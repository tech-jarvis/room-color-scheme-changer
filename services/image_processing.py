import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from configs.constants import FURNITURE_CLASSES, WALL_CLASS

class ImageProcessor:
    def __init__(self, segmentation_model):
        self.seg_model = segmentation_model
    
    def recolor_image(self, image_path: str, palette: List[Tuple[int, int, int]], output_path: str):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segmentation = self.seg_model.segment(Image.fromarray(img))
        
        recolored = img.copy()
        self._recolor_walls(recolored, segmentation, palette[0])
        self._recolor_furniture(recolored, segmentation, palette[1:])
        cv2.imwrite(output_path, cv2.cvtColor(recolored, cv2.COLOR_RGB2BGR))
    
    def _recolor_walls(self, image, segmentation, target_color):
        wall_mask = (segmentation == WALL_CLASS)
        if np.any(wall_mask):
            self._recolor_region(image, wall_mask, np.array(target_color))
    
    def _recolor_furniture(self, image, segmentation, colors):
        furniture_mask = np.isin(segmentation, FURNITURE_CLASSES)
        if np.any(furniture_mask) and len(colors) > 0:
            furniture_pixels = image[furniture_mask]
            kmeans = KMeans(n_clusters=len(colors))
            kmeans.fit(furniture_pixels)
            
            for i, color in enumerate(colors):
                cluster_mask = (kmeans.labels_ == i)
                target_mask = np.zeros_like(furniture_mask)
                target_mask[furniture_mask] = cluster_mask
                self._recolor_region(image, target_mask, np.array(color))
    
    def _recolor_region(self, image, mask, target_color):
        lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        target_lab = cv2.cvtColor(np.uint8([[target_color]]), cv2.COLOR_RGB2LAB)[0][0]
        
        region = lab_img[mask]
        mean, std = region.mean(axis=0), region.std(axis=0)
        new_region = ((region - mean) / std) * np.array([20, 20, 20]) + target_lab
        new_region = np.clip(new_region, 0, 255)
        
        lab_img[mask] = new_region
        image[mask] = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)[mask]