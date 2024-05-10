### Light Source
import numpy as np
from enum import Enum

class LightPosition(Enum):
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    TOP_LEFT = "Top Left Light"
    TOP_RIGHT = "Top Right Light"
    BOTTOM_LEFT = "Bottom Left Light"
    BOTTOM_RIGHT = "Bottom Right Light"

def generate_gradient_image(width:int, height:int, start_color: tuple, end_color: tuple, multiplier: float, lightPosition:LightPosition):
    """
    Generate a gradient image with a light source effect.

    Parameters:
    width (int): Width of the image.
    height (int): Height of the image.
    start_color: Starting color RGB of the gradient.
    end_color: Ending color RGB of the gradient.
    multiplier: Weight of light.
    lightPosition (LightPosition): Position of the light source.

    Returns:
    np.array: 2D gradient image array.
    """
    # Create a gradient from 0 to 1 and apply multiplier
    if lightPosition == LightPosition.LEFT:
        gradient = np.tile(np.linspace(0, 1, width)**multiplier, (height, 1))
    elif lightPosition == LightPosition.RIGHT:
        gradient = np.tile(np.linspace(1, 0, width)**multiplier, (height, 1))
    elif lightPosition == LightPosition.TOP:
        gradient = np.tile(np.linspace(0, 1, height)**multiplier, (width, 1)).T
    elif lightPosition == LightPosition.BOTTOM:
        gradient = np.tile(np.linspace(1, 0, height)**multiplier, (width, 1)).T
    elif lightPosition == LightPosition.BOTTOM_RIGHT:
        x = np.linspace(1, 0, width)**multiplier
        y = np.linspace(1, 0, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    elif lightPosition == LightPosition.BOTTOM_LEFT:
        x = np.linspace(0, 1, width)**multiplier
        y = np.linspace(1, 0, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    elif lightPosition == LightPosition.TOP_RIGHT:
        x = np.linspace(1, 0, width)**multiplier
        y = np.linspace(0, 1, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    elif lightPosition == LightPosition.TOP_LEFT:
        x = np.linspace(0, 1, width)**multiplier
        y = np.linspace(0, 1, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    else:
        raise ValueError(f"Unsupported position. Choose from {', '.join([member.value for member in LightPosition])}.")

    # Interpolate between start_color and end_color based on the gradient
    gradient_img = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(3):
        gradient_img[..., i] = start_color[i] + (end_color[i] - start_color[i]) * gradient
    
    gradient_img = np.clip(gradient_img, 0, 255).astype(np.uint8)
    return gradient_img