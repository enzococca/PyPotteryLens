# metadata_analysis.py

import numpy as np
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
import cv2
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path


@dataclass
class ImageMetadata:
    """Container for image metadata"""
    filename: str
    exif_data: Dict[str, Any]
    file_hash: str
    creation_date: datetime
    modification_date: datetime
    dimensions: Tuple[int, int]
    file_size: int
    provenance: List[Dict[str, Any]]
    tags: List[str]
    relationships: List[Dict[str, str]]
    

class MetadataManager:
    """Manages metadata extraction, preservation, and tracking"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.tag_hierarchy = {
            'typology': {
                'open_forms': ['bowl', 'plate', 'dish', 'cup'],
                'closed_forms': ['jar', 'amphora', 'pitcher', 'bottle'],
                'cooking_ware': ['pot', 'pan', 'lid']
            },
            'decoration': {
                'painted': ['red_figure', 'black_figure', 'geometric'],
                'incised': ['linear', 'wavy', 'crosshatch'],
                'stamped': ['rosette', 'palmette', 'geometric']
            },
            'condition': ['complete', 'fragmentary', 'restored', 'damaged'],
            'period': ['bronze_age', 'iron_age', 'classical', 'hellenistic', 'roman']
        }
    
    def extract_exif(self, image_path: str) -> Dict[str, Any]:
        """Extract and preserve EXIF metadata"""
        try:
            image = Image.open(image_path)
            exifdata = image.getexif()
            
            exif_dict = {}
            for tag_id in exifdata:
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                if isinstance(data, bytes):
                    data = data.decode(errors='ignore')
                exif_dict[tag] = data
            
            return exif_dict
        except Exception as e:
            print(f"Error extracting EXIF: {str(e)}")
            return {}
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file for integrity tracking"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def track_provenance(self, item_id: int, action: str, user: str = "system", 
                        details: Optional[Dict] = None):
        """Track provenance information for an item"""
        provenance_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user": user,
            "details": details or {}
        }
        
        # Store in database
        self.db_manager.add_metadata(
            item_id, 
            {"provenance": json.dumps(provenance_entry)}
        )
    
    def add_hierarchical_tags(self, item_id: int, tags: List[str]):
        """Add tags with hierarchical structure"""
        expanded_tags = set(tags)
        
        # Expand tags based on hierarchy
        for tag in tags:
            for category, values in self.tag_hierarchy.items():
                if isinstance(values, dict):
                    for subcategory, items in values.items():
                        if tag in items:
                            expanded_tags.add(category)
                            expanded_tags.add(subcategory)
                elif isinstance(values, list) and tag in values:
                    expanded_tags.add(category)
        
        # Store tags
        self.db_manager.add_metadata(
            item_id,
            {"tags": json.dumps(list(expanded_tags))}
        )
    
    def define_relationship(self, item1_id: int, item2_id: int, 
                          relationship_type: str, properties: Optional[Dict] = None):
        """Define relationships between pottery items"""
        relationship = {
            "source": item1_id,
            "target": item2_id,
            "type": relationship_type,
            "properties": properties or {},
            "created": datetime.now().isoformat()
        }
        
        # Store bidirectional relationship
        self.db_manager.add_metadata(
            item1_id,
            {f"relationship_{item2_id}": json.dumps(relationship)}
        )
        
        # Reverse relationship
        reverse_rel = relationship.copy()
        reverse_rel["source"] = item2_id
        reverse_rel["target"] = item1_id
        
        self.db_manager.add_metadata(
            item2_id,
            {f"relationship_{item1_id}": json.dumps(reverse_rel)}
        )


class MorphometricAnalyzer:
    """Performs automatic morphometric analysis on pottery profiles"""
    
    def __init__(self):
        self.measurements = {}
    
    def analyze_profile(self, image_path: str) -> Dict[str, float]:
        """Extract morphometric measurements from pottery profile"""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        # Find contours
        edges = cv2.Canny(img, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        # Get largest contour (main pottery profile)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Calculate measurements
        measurements = {}
        
        # Bounding box measurements
        x, y, w, h = cv2.boundingRect(main_contour)
        measurements['height'] = h
        measurements['max_diameter'] = w
        
        # Calculate rim and base diameters
        # Reshape contour to (n_points, 2) for easier indexing
        contour_points = main_contour.reshape(-1, 2)
        
        rim_points = contour_points[contour_points[:, 1] < y + h * 0.1]
        if len(rim_points) > 0:
            rim_width = np.max(rim_points[:, 0]) - np.min(rim_points[:, 0])
            measurements['rim_diameter'] = rim_width
        
        base_points = contour_points[contour_points[:, 1] > y + h * 0.9]
        if len(base_points) > 0:
            base_width = np.max(base_points[:, 0]) - np.min(base_points[:, 0])
            measurements['base_diameter'] = base_width
        
        # Calculate volume approximation (revolution solid)
        measurements['volume_estimate'] = self._estimate_volume(main_contour)
        
        # Shape indices
        measurements['height_diameter_ratio'] = h / w if w > 0 else 0
        measurements['rim_base_ratio'] = (measurements.get('rim_diameter', 0) / 
                                         measurements.get('base_diameter', 1))
        
        # Curvature analysis
        measurements['max_curvature'] = self._calculate_max_curvature(main_contour)
        
        return measurements
    
    def _estimate_volume(self, contour: np.ndarray) -> float:
        """Estimate volume using disk method for solid of revolution"""
        # Get profile points
        points = contour.reshape(-1, 2)
        
        # Sort by y-coordinate
        sorted_points = points[points[:, 1].argsort()]
        
        # Calculate volume using disk method
        volume = 0
        for i in range(len(sorted_points) - 1):
            y1, x1 = sorted_points[i]
            y2, x2 = sorted_points[i + 1]
            
            # Average radius
            r = (x1 + x2) / 2
            
            # Height of disk
            h = abs(y2 - y1)
            
            # Volume of disk
            volume += np.pi * r * r * h
        
        return volume
    
    def _calculate_max_curvature(self, contour: np.ndarray) -> float:
        """Calculate maximum curvature of the profile"""
        points = contour.reshape(-1, 2)
        
        # Calculate curvature using finite differences
        if len(points) < 3:
            return 0
        
        curvatures = []
        for i in range(1, len(points) - 1):
            p1, p2, p3 = points[i-1], points[i], points[i+1]
            
            # Calculate curvature using three points
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p3 - p1)
            
            if a * b * c > 0:
                s = (a + b + c) / 2
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                curvature = 4 * area / (a * b * c)
                curvatures.append(curvature)
        
        return max(curvatures) if curvatures else 0


class PotteryClusterAnalyzer:
    """Performs clustering analysis on pottery collections"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract features for clustering"""
        # Load image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize for consistent feature extraction
        img_resized = cv2.resize(img_rgb, (128, 128))
        
        features = []
        
        # Color histogram features
        for channel in range(3):
            hist = cv2.calcHist([img_resized], [channel], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        # Texture features using LBP
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        lbp = self._calculate_lbp(gray)
        lbp_hist, _ = np.histogram(lbp, bins=32, range=(0, 256))
        features.extend(lbp_hist)
        
        # Shape features using Hu moments
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        features.extend(np.log(np.abs(hu_moments) + 1e-10))
        
        return np.array(features)
    
    def _calculate_lbp(self, image: np.ndarray, radius: int = 1) -> np.ndarray:
        """Calculate Local Binary Pattern"""
        rows, cols = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = image[i, j]
                binary_string = ""
                
                # 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        neighbor = image[i + di, j + dj]
                        binary_string += "1" if neighbor >= center else "0"
                
                lbp[i, j] = int(binary_string, 2)
        
        return lbp
    
    def cluster_pottery(self, features: np.ndarray, method: str = 'kmeans', 
                       n_clusters: int = 5) -> np.ndarray:
        """Cluster pottery based on features"""
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        labels = clusterer.fit_predict(features_scaled)
        
        return labels
    
    def visualize_clusters(self, features: np.ndarray, labels: np.ndarray, 
                          save_path: Optional[str] = None):
        """Visualize clustering results"""
        # Reduce dimensions for visualization
        features_scaled = self.scaler.transform(features)
        features_2d = self.pca.fit_transform(features_scaled)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Pottery Clustering Results')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()


class StatisticalDashboard:
    """Creates statistical visualizations for pottery datasets"""
    
    def __init__(self):
        self.fig_size = (15, 10)
        sns.set_style("whitegrid")
    
    def create_dashboard(self, data: pd.DataFrame, save_path: str):
        """Create comprehensive statistical dashboard"""
        fig = plt.figure(figsize=self.fig_size)
        
        # Layout: 2x3 grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Type distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if 'type' in data.columns:
            type_counts = data['type'].value_counts()
            ax1.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
            ax1.set_title('Pottery Type Distribution')
        
        # 2. Period timeline
        ax2 = fig.add_subplot(gs[0, 1])
        if 'period' in data.columns:
            period_counts = data['period'].value_counts()
            ax2.bar(period_counts.index, period_counts.values)
            ax2.set_title('Distribution by Period')
            ax2.set_xlabel('Period')
            ax2.set_ylabel('Count')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Morphometric scatter
        ax3 = fig.add_subplot(gs[0, 2])
        if all(col in data.columns for col in ['height', 'diameter']):
            ax3.scatter(data['diameter'], data['height'], alpha=0.6)
            ax3.set_xlabel('Diameter (mm)')
            ax3.set_ylabel('Height (mm)')
            ax3.set_title('Height vs Diameter')
        
        # 4. Volume distribution
        ax4 = fig.add_subplot(gs[1, 0])
        if 'volume' in data.columns:
            ax4.hist(data['volume'], bins=20, edgecolor='black')
            ax4.set_xlabel('Volume (ml)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Volume Distribution')
        
        # 5. Heatmap of relationships
        ax5 = fig.add_subplot(gs[1, 1])
        if 'type' in data.columns and 'period' in data.columns:
            pivot_table = pd.crosstab(data['type'], data['period'])
            sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlOrRd', ax=ax5)
            ax5.set_title('Type-Period Correlation')
        
        # 6. Trend analysis
        ax6 = fig.add_subplot(gs[1, 2])
        if 'date_added' in data.columns:
            data['date_added'] = pd.to_datetime(data['date_added'])
            daily_counts = data.groupby(data['date_added'].dt.date).size()
            ax6.plot(daily_counts.index, daily_counts.values)
            ax6.set_title('Data Collection Timeline')
            ax6.set_xlabel('Date')
            ax6.set_ylabel('Items Added')
            plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle('Pottery Collection Statistical Dashboard', fontsize=16)
        # Use constrained_layout instead of tight_layout to avoid warnings
        try:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        except:
            # Fallback if tight_layout fails
            pass
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_morphometric_report(self, measurements: List[Dict], save_path: str):
        """Generate detailed morphometric analysis report"""
        df = pd.DataFrame(measurements)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Height distribution
        if 'height' in df.columns:
            axes[0, 0].hist(df['height'], bins=15, edgecolor='black')
            axes[0, 0].set_xlabel('Height (mm)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Height Distribution')
            axes[0, 0].axvline(df['height'].mean(), color='red', 
                              linestyle='dashed', linewidth=2, label='Mean')
            axes[0, 0].legend()
        
        # Diameter distribution
        if 'max_diameter' in df.columns:
            axes[0, 1].hist(df['max_diameter'], bins=15, edgecolor='black')
            axes[0, 1].set_xlabel('Max Diameter (mm)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Diameter Distribution')
        
        # Ratio analysis
        if 'height_diameter_ratio' in df.columns:
            axes[1, 0].scatter(df.index, df['height_diameter_ratio'])
            axes[1, 0].set_xlabel('Item Index')
            axes[1, 0].set_ylabel('H/D Ratio')
            axes[1, 0].set_title('Height/Diameter Ratio')
            axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.5)
        
        # Volume vs dimensions
        if all(col in df.columns for col in ['volume_estimate', 'height', 'max_diameter']):
            volume_ratio = df['volume_estimate'] / (df['height'] * df['max_diameter'])
            axes[1, 1].hist(volume_ratio, bins=15, edgecolor='black')
            axes[1, 1].set_xlabel('Volume Coefficient')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Shape Complexity')
        
        plt.suptitle('Morphometric Analysis Report', fontsize=14)
        # Use constrained_layout instead of tight_layout to avoid warnings
        try:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        except:
            # Fallback if tight_layout fails
            pass
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()