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

def generate_gradient_image(width:int, height:int, lightPosition:LightPosition):
    """
    Generate a gradient image with a light source effect.
    
    Parameters:
    width (int): Width of the image.
    height (int): Height of the image.
    lightPosition (str): Position of the light source. 
                     It can be 'Left Light', 'Right Light', 'Top Light', 'Bottom Light',
                     'Top Left Light', 'Top Right Light', 'Bottom Left Light', 'Bottom Right Light'.
    
    Returns:
    np.array: 2D gradient image array.
    """
    if lightPosition == LightPosition.LEFT:
        gradient = np.tile(np.linspace(255, 0, width), (height, 1))
    elif lightPosition == LightPosition.RIGHT:
        gradient = np.tile(np.linspace(0, 255, width), (height, 1))
    elif lightPosition == LightPosition.TOP:
        gradient = np.tile(np.linspace(255, 0, height), (width, 1)).T
    elif lightPosition == LightPosition.BOTTOM:
        gradient = np.tile(np.linspace(0, 255, height), (width, 1)).T
    elif lightPosition == LightPosition.TOP_LEFT:
        x = np.linspace(255, 0, width)
        y = np.linspace(255, 0, height)
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = (x_mesh + y_mesh) / 2
    elif lightPosition == LightPosition.TOP_RIGHT:
        x = np.linspace(0, 255, width)
        y = np.linspace(255, 0, height)
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = (x_mesh + y_mesh) / 2
    elif lightPosition == LightPosition.BOTTOM_LEFT:
        x = np.linspace(255, 0, width)
        y = np.linspace(0, 255, height)
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = (x_mesh + y_mesh) / 2
    elif lightPosition == LightPosition.BOTTOM_RIGHT:
        x = np.linspace(0, 255, width)
        y = np.linspace(0, 255, height)
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = (x_mesh + y_mesh) / 2
    else:
        raise ValueError("Unsupported position. Choose from 'Left Light', 'Right Light', 'Top Light', 'Bottom Light','Top Left Light', 'Top Right Light', 'Bottom Left Light', 'Bottom Right Light'.")
    
    gradient = np.stack((gradient,) * 3, axis=-1).astype(np.uint8)

    return gradient