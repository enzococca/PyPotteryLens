# cidoc_crm_export.py

import json
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid
from dataclasses import dataclass, field


@dataclass
class CRMEntity:
    """Base class for CIDOC-CRM entities"""
    uri: str
    type: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    

class CIDOCCRMExporter:
    """Export pottery data to CIDOC-CRM standard format"""
    
    def __init__(self):
        self.namespace = {
            'crm': 'http://www.cidoc-crm.org/cidoc-crm/',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'xsd': 'http://www.w3.org/2001/XMLSchema#'
        }
        
        # CIDOC-CRM class mappings for pottery
        self.crm_classes = {
            'pottery_item': 'E22_Human-Made_Object',
            'production': 'E12_Production',
            'type_assignment': 'E17_Type_Assignment',
            'measurement': 'E16_Measurement',
            'place': 'E53_Place',
            'time_span': 'E52_Time-Span',
            'person': 'E21_Person',
            'material': 'E57_Material',
            'technique': 'E29_Design_or_Procedure',
            'documentation': 'E31_Document'
        }
        
        # CIDOC-CRM property mappings
        self.crm_properties = {
            'has_type': 'P2_has_type',
            'carries_out': 'P14_carried_out_by',
            'has_dimension': 'P43_has_dimension',
            'consists_of': 'P45_consists_of',
            'has_time_span': 'P4_has_time-span',
            'took_place_at': 'P7_took_place_at',
            'used_technique': 'P32_used_general_technique',
            'documents': 'P70_documents',
            'has_current_location': 'P55_has_current_location',
            'has_former_location': 'P53_has_former_or_current_location'
        }
    
    def create_pottery_entity(self, item_data: Dict) -> CRMEntity:
        """Create a CIDOC-CRM entity for a pottery item"""
        entity_id = f"pottery_{uuid.uuid4().hex[:8]}"
        
        entity = CRMEntity(
            uri=f"http://example.org/pottery/{entity_id}",
            type=self.crm_classes['pottery_item'],
            label=item_data.get('filename', 'Unknown pottery item')
        )
        
        # Add type information
        if 'type' in item_data:
            entity.properties[self.crm_properties['has_type']] = {
                'value': item_data['type'],
                'vocabulary': 'pottery_typology'
            }
        
        # Add dimensional information
        if any(k in item_data for k in ['height', 'diameter', 'thickness']):
            measurements = []
            
            if 'height' in item_data:
                measurements.append({
                    'dimension': 'height',
                    'value': item_data['height'],
                    'unit': 'mm'
                })
            
            if 'diameter' in item_data:
                measurements.append({
                    'dimension': 'diameter',
                    'value': item_data['diameter'],
                    'unit': 'mm'
                })
            
            entity.properties[self.crm_properties['has_dimension']] = measurements
        
        # Add production information
        if 'production_date' in item_data or 'production_place' in item_data:
            production = {
                'type': self.crm_classes['production'],
                'uri': f"http://example.org/production/{entity_id}"
            }
            
            if 'production_date' in item_data:
                production['time_span'] = {
                    'earliest': item_data.get('date_earliest'),
                    'latest': item_data.get('date_latest'),
                    'label': item_data['production_date']
                }
            
            if 'production_place' in item_data:
                production['place'] = {
                    'label': item_data['production_place'],
                    'coordinates': item_data.get('coordinates')
                }
            
            entity.properties['production'] = production
        
        return entity
    
    def export_to_rdf_xml(self, entities: List[CRMEntity], output_path: str):
        """Export entities to RDF/XML format"""
        root = ET.Element('rdf:RDF')
        
        # Add namespaces
        for prefix, uri in self.namespace.items():
            root.set(f'xmlns:{prefix}', uri)
        
        for entity in entities:
            # Create entity element
            entity_elem = ET.SubElement(root, f'crm:{entity.type}')
            entity_elem.set('rdf:about', entity.uri)
            
            # Add label
            label_elem = ET.SubElement(entity_elem, 'rdfs:label')
            label_elem.text = entity.label
            
            # Add properties
            for prop_name, prop_value in entity.properties.items():
                self._add_property_to_xml(entity_elem, prop_name, prop_value)
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    def export_to_json_ld(self, entities: List[CRMEntity], output_path: str):
        """Export entities to JSON-LD format"""
        context = {
            "@context": {
                "crm": "http://www.cidoc-crm.org/cidoc-crm/",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "label": "rdfs:label",
                "@base": "http://example.org/pottery/"
            }
        }
        
        graph = []
        
        for entity in entities:
            json_entity = {
                "@id": entity.uri,
                "@type": f"crm:{entity.type}",
                "label": entity.label
            }
            
            # Add properties
            for prop_name, prop_value in entity.properties.items():
                json_entity[prop_name] = self._convert_property_to_json(prop_value)
            
            graph.append(json_entity)
        
        output = {**context, "@graph": graph}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    def _add_property_to_xml(self, parent: ET.Element, prop_name: str, prop_value: Any):
        """Add property to XML element"""
        if isinstance(prop_value, dict):
            prop_elem = ET.SubElement(parent, f'crm:{prop_name}')
            for key, value in prop_value.items():
                sub_elem = ET.SubElement(prop_elem, f'crm:{key}')
                sub_elem.text = str(value)
        elif isinstance(prop_value, list):
            for item in prop_value:
                self._add_property_to_xml(parent, prop_name, item)
        else:
            prop_elem = ET.SubElement(parent, f'crm:{prop_name}')
            prop_elem.text = str(prop_value)
    
    def _convert_property_to_json(self, prop_value: Any) -> Any:
        """Convert property value to JSON-compatible format"""
        if isinstance(prop_value, dict):
            return {k: self._convert_property_to_json(v) for k, v in prop_value.items()}
        elif isinstance(prop_value, list):
            return [self._convert_property_to_json(item) for item in prop_value]
        else:
            return str(prop_value)
    
    def create_archaeological_context(self, context_data: Dict) -> Dict:
        """Create archaeological context following CIDOC-CRM"""
        context = {
            "@type": "crm:E53_Place",
            "@id": f"context_{uuid.uuid4().hex[:8]}",
            "label": context_data.get('name', 'Unknown context'),
            "coordinates": {
                "@type": "crm:E47_Spatial_Coordinates",
                "latitude": context_data.get('latitude'),
                "longitude": context_data.get('longitude'),
                "elevation": context_data.get('elevation'),
                "coordinate_system": "WGS84"
            }
        }
        
        # Add stratigraphic information
        if 'stratum' in context_data:
            context['stratigraphic_unit'] = {
                "@type": "crm:E18_Physical_Thing",
                "label": context_data['stratum'],
                "has_type": "stratigraphic_unit"
            }
        
        # Add temporal information
        if 'period' in context_data:
            context['has_time_span'] = {
                "@type": "crm:E52_Time-Span",
                "label": context_data['period'],
                "earliest": context_data.get('period_start'),
                "latest": context_data.get('period_end')
            }
        
        return context