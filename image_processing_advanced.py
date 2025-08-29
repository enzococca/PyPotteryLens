# image_processing_advanced.py

import numpy as np
from PIL import Image, ImageEnhance
import cv2
from typing import Tuple, Optional, List, Union
import torch
import torchvision.transforms as transforms
from dataclasses import dataclass
from pathlib import Path
from sklearn.cluster import KMeans
from skimage import exposure, color, restoration
import sqlite3
from datetime import datetime
import json
from functools import lru_cache
import hashlib


@dataclass
class ImageEnhancementConfig:
    """Configuration for image enhancement operations"""
    color_correction: bool = True
    histogram_equalization: bool = True
    denoise: bool = True
    sharpen: bool = True
    target_brightness: float = 0.5
    target_contrast: float = 1.2
    
    
class ColorNormalizer:
    """Handles color normalization for pottery images"""
    
    def __init__(self, reference_stats: Optional[dict] = None):
        self.reference_stats = reference_stats or {
            'mean': [0.7, 0.65, 0.6],  # Typical pottery color
            'std': [0.15, 0.15, 0.15]
        }
        
    def normalize_color(self, image: np.ndarray) -> np.ndarray:
        """Normalize image colors to match reference statistics"""
        if len(image.shape) == 2:
            # Convert grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Convert to LAB color space for better color manipulation
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Normalize L channel
        l = exposure.equalize_adapthist(l, clip_limit=0.03)
        l = (l * 255).astype(np.uint8)
        
        # Reduce color cast - ensure we maintain array dimensions
        a = np.clip(a.astype(np.float32), -10, 10).astype(np.uint8)
        b = np.clip(b.astype(np.float32), -10, 10).astype(np.uint8)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return result
    
    def auto_white_balance(self, image: np.ndarray) -> np.ndarray:
        """Apply automatic white balance"""
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Gray world assumption
        avg_r = np.mean(image[:, :, 0])
        avg_g = np.mean(image[:, :, 1])
        avg_b = np.mean(image[:, :, 2])
        
        avg_gray = (avg_r + avg_g + avg_b) / 3
        
        # Calculate scaling factors
        scale_r = avg_gray / avg_r
        scale_g = avg_gray / avg_g
        scale_b = avg_gray / avg_b
        
        # Apply scaling
        balanced = image.copy()
        balanced[:, :, 0] = np.clip(image[:, :, 0] * scale_r, 0, 255)
        balanced[:, :, 1] = np.clip(image[:, :, 1] * scale_g, 0, 255)
        balanced[:, :, 2] = np.clip(image[:, :, 2] * scale_b, 0, 255)
        
        return balanced.astype(np.uint8)
    

class WatermarkRemover:
    """Removes watermarks and stamps from pottery images"""
    
    def __init__(self):
        self.inpainting_radius = 3
        
    def detect_watermark_region(self, image: np.ndarray) -> np.ndarray:
        """Detect potential watermark regions using frequency analysis"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # High-pass filter to detect repetitive patterns
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        r = 30
        center = (crow, ccol)
        cv2.circle(mask, center, r, 0, -1)
        
        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Threshold to get potential watermark mask
        _, watermark_mask = cv2.threshold(
            img_back.astype(np.uint8), 
            np.percentile(img_back, 95), 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        watermark_mask = cv2.morphologyEx(watermark_mask, cv2.MORPH_CLOSE, kernel)
        watermark_mask = cv2.morphologyEx(watermark_mask, cv2.MORPH_OPEN, kernel)
        
        return watermark_mask
    
    def remove_watermark(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Remove watermark using inpainting"""
        if mask is None:
            mask = self.detect_watermark_region(image)
        
        # Inpaint the watermark region
        result = cv2.inpaint(
            image, 
            mask, 
            self.inpainting_radius, 
            cv2.INPAINT_TELEA
        )
        
        return result


class ImageComparator:
    """Handles side-by-side comparison of pottery images"""
    
    def __init__(self):
        self.sync_zoom_level = 1.0
        self.sync_pan_offset = (0, 0)
        
    def create_comparison_view(
        self, 
        images: List[np.ndarray], 
        labels: Optional[List[str]] = None,
        layout: str = 'horizontal'
    ) -> np.ndarray:
        """Create a comparison view of multiple images"""
        if not images:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Convert all images to RGB to ensure consistency
        rgb_images = []
        for img in images:
            if len(img.shape) == 2:  # Grayscale
                rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
                rgb_img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 3:  # Already RGB
                rgb_img = img
            else:
                # Handle unexpected formats by converting to RGB
                if len(img.shape) == 3 and img.shape[2] == 1:
                    # Single channel image in 3D format
                    rgb_img = cv2.cvtColor(img[:,:,0], cv2.COLOR_GRAY2RGB)
                else:
                    # Default case - assume RGB
                    rgb_img = img
            rgb_images.append(rgb_img)
        
        # Ensure all images have the same height
        target_height = max(img.shape[0] for img in rgb_images)
        resized_images = []
        
        for img in rgb_images:
            if img.shape[0] != target_height:
                scale = target_height / img.shape[0]
                new_width = int(img.shape[1] * scale)
                resized = cv2.resize(img, (new_width, target_height))
                resized_images.append(resized)
            else:
                resized_images.append(img)
        
        # Add labels if provided
        if labels:
            labeled_images = []
            for img, label in zip(resized_images, labels):
                # Add label bar
                label_height = 30
                label_bar = np.ones((label_height, img.shape[1], 3), dtype=np.uint8) * 200
                cv2.putText(
                    label_bar, label, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
                )
                labeled_img = np.vstack([label_bar, img])
                labeled_images.append(labeled_img)
            resized_images = labeled_images
        
        # Concatenate images
        if layout == 'horizontal':
            comparison = np.hstack(resized_images)
        else:  # vertical
            comparison = np.vstack(resized_images)
        
        return comparison
    
    def create_overlay_view(
        self, 
        images: List[np.ndarray], 
        alphas: Optional[List[float]] = None
    ) -> np.ndarray:
        """Create transparent overlay of multiple images"""
        if not images:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        if alphas is None:
            alphas = [1.0 / len(images)] * len(images)
        
        # Convert all images to RGB to ensure consistency
        rgb_images = []
        for img in images:
            if len(img.shape) == 2:  # Grayscale
                rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
                rgb_img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 3:  # Already RGB
                rgb_img = img
            else:
                # Handle unexpected formats by converting to RGB
                if len(img.shape) == 3 and img.shape[2] == 1:
                    # Single channel image in 3D format
                    rgb_img = cv2.cvtColor(img[:,:,0], cv2.COLOR_GRAY2RGB)
                else:
                    # Default case - assume RGB
                    rgb_img = img
            rgb_images.append(rgb_img)
        
        # Ensure all images have the same dimensions
        target_shape = rgb_images[0].shape
        result = np.zeros(target_shape, dtype=np.float32)
        
        for img, alpha in zip(rgb_images, alphas):
            if img.shape != target_shape:
                img = cv2.resize(img, (target_shape[1], target_shape[0]))
            result += img.astype(np.float32) * alpha
        
        return np.clip(result, 0, 255).astype(np.uint8)


class DatabaseManager:
    """Manages SQLite database for pottery data"""
    
    def __init__(self, db_path: str = "pypotterylens.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pottery_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                source_pdf TEXT,
                source_folder TEXT,
                page_number INTEGER,
                instance_number INTEGER,
                type TEXT,
                position TEXT,
                rotation TEXT,
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Check if source_folder column exists, add it if missing (migration)
        cursor.execute("PRAGMA table_info(pottery_items)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'source_folder' not in columns:
            cursor.execute('ALTER TABLE pottery_items ADD COLUMN source_folder TEXT')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pottery_id INTEGER,
                user TEXT,
                annotation_type TEXT,
                annotation_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (pottery_id) REFERENCES pottery_items(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pottery_id INTEGER,
                key TEXT,
                value TEXT,
                FOREIGN KEY (pottery_id) REFERENCES pottery_items(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_hash TEXT UNIQUE,
                operation TEXT,
                result_path TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_pottery_item(self, item_data: dict) -> int:
        """Add a pottery item to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO pottery_items 
            (filename, source_pdf, source_folder, page_number, instance_number, type, position, rotation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item_data['filename'],
            item_data.get('source_pdf'),
            item_data.get('source_folder'),
            item_data.get('page_number'),
            item_data.get('instance_number'),
            item_data.get('type'),
            item_data.get('position'),
            item_data.get('rotation')
        ))
        
        item_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return item_id
    
    def query_items(self, filters: dict = None) -> List[dict]:
        """Query pottery items with optional filters"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM pottery_items WHERE 1=1"
        params = []
        
        if filters:
            for key, value in filters.items():
                query += f" AND {key} = ?"
                params.append(value)
        
        cursor.execute(query, params)
        items = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return items
    
    def add_metadata(self, pottery_id: int, metadata: dict):
        """Add metadata to a pottery item"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for key, value in metadata.items():
            cursor.execute('''
                INSERT OR REPLACE INTO metadata (pottery_id, key, value)
                VALUES (?, ?, ?)
            ''', (pottery_id, key, str(value)))
        
        conn.commit()
        conn.close()


class PerformanceOptimizer:
    """Handles caching and performance optimization"""
    
    def __init__(self, cache_dir: str = "cache", max_cache_size_mb: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size = max_cache_size_mb * 1024 * 1024
        self.db_manager = DatabaseManager()
        
    @lru_cache(maxsize=128)
    def get_cached_result(self, operation: str, input_hash: str) -> Optional[str]:
        """Get cached result for an operation"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT result_path FROM processing_cache
            WHERE input_hash = ? AND operation = ?
        ''', (input_hash, operation))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def cache_result(self, operation: str, input_data: Union[np.ndarray, str], result_path: str):
        """Cache processing result"""
        # Generate hash of input
        if isinstance(input_data, np.ndarray):
            input_hash = hashlib.md5(input_data.tobytes()).hexdigest()
        else:
            input_hash = hashlib.md5(str(input_data).encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO processing_cache
            (input_hash, operation, result_path)
            VALUES (?, ?, ?)
        ''', (input_hash, operation, result_path))
        
        conn.commit()
        conn.close()
        
        # Check cache size and clean if needed
        self._check_cache_size()
    
    def _check_cache_size(self):
        """Check and clean cache if it exceeds size limit"""
        total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
        
        if total_size > self.max_cache_size:
            # Remove oldest cached files
            files = [(f, f.stat().st_mtime) for f in self.cache_dir.rglob('*') if f.is_file()]
            files.sort(key=lambda x: x[1])
            
            while total_size > self.max_cache_size * 0.8 and files:
                file, _ = files.pop(0)
                size = file.stat().st_size
                file.unlink()
                total_size -= size


class GISExporter:
    """Handles export to GIS formats"""
    
    def export_to_geojson(self, items: List[dict], output_path: str):
        """Export pottery items to GeoJSON format"""
        features = []
        
        for item in items:
            # Create feature
            feature = {
                "type": "Feature",
                "properties": {
                    "id": item.get('id'),
                    "filename": item.get('filename'),
                    "type": item.get('type'),
                    "position": item.get('position'),
                    "rotation": item.get('rotation'),
                    "source_pdf": item.get('source_pdf'),
                    "page_number": item.get('page_number'),
                    "instance_number": item.get('instance_number')
                },
                "geometry": {
                    "type": "Point",
                    # Default coordinates - should be updated with actual location
                    "coordinates": [0.0, 0.0]
                }
            }
            
            # Add any additional metadata
            if 'metadata' in item:
                feature['properties'].update(item['metadata'])
            
            features.append(feature)
        
        # Create GeoJSON structure
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
                }
            }
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
    
    def export_to_shapefile(self, items: List[dict], output_path: str):
        """Export pottery items to Shapefile format (requires geopandas)"""
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            
            # Create GeoDataFrame
            geometry = [Point(0, 0) for _ in items]  # Default location
            gdf = gpd.GeoDataFrame(items, geometry=geometry)
            
            # Save to shapefile
            gdf.to_file(output_path, driver='ESRI Shapefile')
            
        except ImportError:
            print("Geopandas not installed. Install with: pip install geopandas")
            return False
        
        return True