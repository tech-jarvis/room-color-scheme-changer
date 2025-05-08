import numpy as np
import cv2
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from typing import List, Tuple

def rgb_to_lab(rgb_color: Tuple[int, int, int]) -> LabColor:
    rgb = sRGBColor(rgb_color[0], rgb_color[1], rgb_color[2], is_upscaled=True)
    return convert_color(rgb, LabColor)

def find_matching_colors(palette: List[Tuple[int, int, int]],
                       candidate_colors: List[Tuple[int, int, int]],
                       num_colors: int) -> List[Tuple[int, int, int]]:
    if not candidate_colors:
        return []
    
    avg_color = np.mean([color for color in palette], axis=0)
    avg_lab = rgb_to_lab(avg_color)
    
    diffs = []
    for color in candidate_colors:
        color_lab = rgb_to_lab(color)
        diffs.append(delta_e_cie2000(avg_lab, color_lab))
    
    return [color for _, color in sorted(zip(diffs, candidate_colors))][:num_colors]