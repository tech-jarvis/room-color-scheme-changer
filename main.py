import argparse
from models.segmentation import SegmentationModel
from services.color_analysis import ColorAnalyzer
from services.image_processing import ImageProcessor
from utils.visualization import display_palettes

def main():
    parser = argparse.ArgumentParser(description='Room Color Palette Generator')
    parser.add_argument('image_path', type=str, help='Path to room image')
    args = parser.parse_args()
    
    # Initialize components
    seg_model = SegmentationModel()
    color_analyzer = ColorAnalyzer(seg_model)
    image_processor = ImageProcessor(seg_model)
    
    # Process image
    furniture_colors = color_analyzer.extract_furniture_colors(args.image_path)
    wall_color = color_analyzer.extract_wall_color(args.image_path)
    palettes = color_analyzer.generate_color_palettes(furniture_colors, wall_color)
    
    # Display results
    display_palettes(palettes, args.image_path)
    
    # Example: Apply first palette
    image_processor.recolor_image(args.image_path, palettes[0], "recolored_room.jpg")

if __name__ == "__main__":
    main()