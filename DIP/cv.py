from typing import List, Union, Literal
from DIP.helper import Matrix
from math import log
def calculate_glcm(
    matrix: Matrix, 
    distance: int, 
    theta: Literal[0, 45, 90, 135], 
    normalize: bool = False
) -> Union[List[List[int]], List[List[float]]]:
    """
    Calculate the Symmetric Gray-Level Co-occurrence Matrix (GLCM) for a given grayscale matrix.
    
    Parameters:
        `matrix` (Matrix): 2D grayscale matrix 
        `distance` (int): Distance between pixels to consider
        `theta` (Literal[0, 45, 90, 135]): Angle in degrees for pixel pair direction
        `normalize` (bool, optional): Whether to normalize the GLCM. Defaults to False.
    
    Returns:
        Union[List[List[int]], List[List[float]]]: Symmetric GLCM matrix (either integer or float)
    """
    # Find the maximum gray level in the matrix to determine GLCM size
    max_gray_level = max(max(row) for row in matrix)
    glcm_size = max_gray_level + 1
    
    # Initialize GLCM matrix with zeros
    glcm = [[0 for _ in range(glcm_size)] for _ in range(glcm_size)]
    
    # Define offset calculations for each theta, scaled by distance
    angle_offsets = {
        0: (distance, 0),     # 0 degrees: horizontal
        45: (distance, -distance),    # 45 degrees: diagonal up-right
        90: (0, distance),    # 90 degrees: vertical
        135: (distance, distance)   # 135 degrees: diagonal down-right
    }
    
    # Get the offset based on the specified theta
    dx, dy = angle_offsets[theta]
    
    # Iterate through the matrix to count co-occurrences
    height = len(matrix)
    width = len(matrix[0])
    
    for y in range(height):
        for x in range(width):
            # Check if the offset pixel is within the matrix bounds
            new_x = x + dx
            new_y = y + dy
            
            if 0 <= new_x < width and 0 <= new_y < height:
                # Get current pixel and offset pixel values
                current_pixel = matrix[y][x]
                offset_pixel = matrix[new_y][new_x]
                
                # Symmetric GLCM: increment both symmetric positions
                if current_pixel != offset_pixel:
                    # For different gray levels, increment both symmetric positions
                    glcm[current_pixel][offset_pixel] += 1
                    glcm[offset_pixel][current_pixel] += 1
                else:
                    # For same gray level (diagonal), multiply by 2
                    glcm[current_pixel][offset_pixel] += 2
    
    # Normalize if requested
    if normalize:
        # Calculate total sum for normalization
        total_sum = sum(sum(row) for row in glcm)
        
        # Create a new normalized GLCM
        normalized_glcm = [
            [round(count / total_sum,3) for count in row] 
            for row in glcm
        ]
        
        return normalized_glcm
    
    return glcm

def agg_glcm(glcm: List[List[float]]) -> tuple[float, float, float]:
    """
    Calculate Contrast, Homogeneity, and Entropy from a given normalized Gray Level Co-occurrence Matrix (GLCM).

    Parameters:
        glcm (list of list of float): The GLCM matrix where glcm[i][j] represents 
                                      the co-occurrence probability of intensity levels i and j.
    
    Returns:
        tuple: A tuple containing:
            - Contrast (float): Measures the intensity contrast between a pixel and its neighbor 
                                over the whole image. Higher values indicate larger intensity differences.
            - Homogeneity (float): Measures the closeness of the distribution of elements in the GLCM 
                                   to the diagonal. Higher values indicate a more uniform texture.
            - Entropy (float): Measures the randomness of the intensity distribution in the GLCM. 
                               Higher values indicate more complex textures.

    Definitions:
        - Contrast: 
          Formula: sum((i - j)^2 * P(i, j)) for all i, j.
          Interpretation: High contrast means the image has sharp intensity differences.

        - Homogeneity:
          Formula: sum(P(i, j) / (1 + (i - j)^2)) for all i, j.
          Interpretation: High homogeneity means the texture is uniform.

        - Entropy:
          Formula: -sum(P(i, j) * log(P(i, j))) for all i, j.
          Interpretation: High entropy means the image texture is more complex or random.

    Example:
        glcm = [
            [0.1, 0.2, 0.0],
            [0.2, 0.4, 0.1],
            [0.0, 0.1, 0.0]
        ]
        contrast, homogeneity, entropy = agg_glcm(glcm)
    """
    contrast = 0
    homogeneity = 0
    entropy = 0

    # Calculate dimensions of the GLCM
    size = len(glcm)

    for i in range(size):
        for j in range(size):
            # Get the value at position (i, j)
            p = glcm[i][j]

            # Contrast calculation
            contrast += (i - j) ** 2 * p

            # Homogeneity calculation
            homogeneity += p / (1 + (i - j)**2)

            # Entropy calculation (only if p > 0 to avoid log(0))
            if p > 0:
                entropy -= p * log(p)

    return round(contrast, 3), round(homogeneity, 3), round(entropy, 3)

