import cv2
import numpy as np
from sklearn.cluster import KMeans
from colorthief import ColorThief
from typing import List, Tuple
from configs.constants import FURNITURE_CLASSES, WALL_CLASS, DOMINANT_COLORS_NUM
from utils.color_utils import rgb_to_lab, find_matching_colors

class ColorAnalyzer:
    def __init__(self, segmentation_model):
        self.seg_model = segmentation_model
    
    def extract_furniture_colors(self, image_path: str) -> List[Tuple[int, int, int]]:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        segmentation = self.seg_model.segment(img)
        
        furniture_mask = np.isin(segmentation, FURNITURE_CLASSES)
        furniture_pixels = img_np[furniture_mask]
        
        if len(furniture_pixels) == 0:
            furniture_pixels = img_np.reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=DOMINANT_COLORS_NUM)
        kmeans.fit(furniture_pixels)
        return [tuple(color) for color in kmeans.cluster_centers_.astype(int)]
    
    def extract_wall_color(self, image_path: str) -> Tuple[int, int, int]:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        segmentation = self.seg_model.segment(img)
        
        wall_mask = (segmentation == WALL_CLASS)
        wall_pixels = img_np[wall_mask]
        
        if len(wall_pixels) == 0:
            return ColorThief(image_path).get_color(quality=1)
        
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(wall_pixels)
        return tuple(kmeans.cluster_centers_[0].astype(int))
    
    def generate_color_palettes(self, base_colors: List[Tuple[int, int, int]], 
                              wall_color: Tuple[int, int, int]]) -> List[List[Tuple[int, int, int]]]:
        palettes = []
        palettes.append(self._generate_analogous(wall_color, base_colors))
        palettes.append(self._generate_complementary(wall_color, base_colors))
        palettes.append(self._generate_triadic(wall_color, base_colors))
        palettes.append(self._generate_monochromatic(wall_color, base_colors))
        palettes.append(self._generate_tetradic(wall_color, base_colors))
        return palettes
    
    def _generate_analogous(self, base_color, furniture_colors):
        hsv = cv2.cvtColor(np.uint8([[base_color]]), cv2.COLOR_RGB2HSV)[0][0]
        palette = [base_color]
        for i in range(1, 3):
            new_hue = (hsv[0] + 30 * i) % 180
            new_color = cv2.cvtColor(np.uint8([[[new_hue, hsv[1], hsv[2]]]]), cv2.COLOR_HSV2RGB)[0][0]
            palette.append(tuple(new_color))
        palette.extend(find_matching_colors(palette, furniture_colors, 2))
        return palette
    
    # Implement other palette generation methods (_generate_complementary, etc.)
    # ...