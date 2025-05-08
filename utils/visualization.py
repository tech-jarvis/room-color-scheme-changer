import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

def display_palettes(palettes: List[List[Tuple[int, int, int]]], 
                   original_image_path: str,
                   palette_names=None):
    if palette_names is None:
        palette_names = ["Analogous", "Complementary", "Triadic", "Monochromatic", "Tetradic"]
    
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 2, 1)
    plt.imshow(plt.imread(original_image_path))
    plt.title("Original Room")
    plt.axis('off')
    
    for i, (palette, name) in enumerate(zip(palettes, palette_names), 1):
        plt.subplot(3, 2, i+1)
        plot_palette(palette)
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_palette(palette: List[Tuple[int, int, int]]):
    palette_array = np.array(palette).reshape(1, len(palette), 3)
    plt.imshow(palette_array.astype('uint8'))