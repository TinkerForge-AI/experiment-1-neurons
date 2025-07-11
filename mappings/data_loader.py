import numpy as np
import os

def load_pixel(rgb_tuple):
    """
    Loads a single pixel's RGB value as input.
    
    Args:
        rgb_tuple: RGB values as tuple (r, g, b) where each component is 0-255
        
    Returns:
        numpy array of RGB values
    """
    return np.array(rgb_tuple, dtype=np.uint8)

def load_pixel_from_file(filepath):
    """
    Load pixel data from a .npy file.
    
    Args:
        filepath: Path to .npy file containing pixel data
        
    Returns:
        numpy array of pixel data
    """
    if os.path.exists(filepath):
        return np.load(filepath)
    else:
        raise FileNotFoundError(f"Pixel data file not found: {filepath}")

def normalize_pixel(rgb_values):
    """
    Normalize RGB values to 0-1 range.
    
    Args:
        rgb_values: RGB values (0-255 range)
        
    Returns:
        Normalized RGB values (0-1 range)
    """
    return np.array(rgb_values) / 255.0

def extract_color_component(rgb_values, component='green'):
    """
    Extract specific color component from RGB values.
    
    Args:
        rgb_values: RGB tuple or array
        component: 'red', 'green', or 'blue'
        
    Returns:
        Single color component value
    """
    rgb_array = np.array(rgb_values)
    component_map = {'red': 0, 'green': 1, 'blue': 2}
    
    if component not in component_map:
        raise ValueError(f"Invalid component '{component}'. Use 'red', 'green', or 'blue'.")
    
    return rgb_array[component_map[component]]