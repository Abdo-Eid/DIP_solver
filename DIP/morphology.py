from typing import Generator,Tuple
from DIP.helper import Matrix, slice_matrix, filter_by_kernel, structuring_element_to_set, pixel_coordinates, intersection, union
import DIP.neighborhood_operations as DN
import DIP.operations as DO

def practical_dilation(matrix: Matrix, se: Matrix) -> Matrix:
    """
    Perform dilation operation on a matrix using a given structuring element (se).

    Parameters:
        `matrix` (Matrix): The original binary matrix on which the dilation is applied.
        `se` (Matrix): The structuring element used for the dilation operation.

    Returns:
        Matrix: The result of the dilation operation on the matrix.

    Description:
        The dilation process involves sliding the `se` over the matrix and checking if any '1' pixel in the `se` aligns with a '1' pixel in the matrix neighborhood.
        If such an overlap is found, the center pixel of the neighborhood is set to '1' in the output matrix; otherwise, it remains '0'.
        The output matrix has the same size as the input matrix.
    
    Examples:
        >>> matrix = [[1, 0, 0, 0],
        ...           [0, 0, 0, 0],
        ...           [0, 0, 0, 0],
        ...           [0, 0, 0, 1]]
        >>> se = [[1, 1],
        ...       [1, 1]]
        >>> result = dilation(matrix, se)
        >>> result
        [[1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1]]
    """
    height = len(matrix)
    width = len(matrix[0])

    se_h = len(se)
    se_w = len(se[0])

    # output will be same size as original in this case
    output_mat = [[0 for _ in range(width)] for _ in range(height)]

    # make bigger padded matrix 
    matrix = DN.pad(matrix,(se_w,se_h))

    for i in range(height):
        for j in range(width):
            window = slice_matrix(matrix,(i,i+se_h), (j,j+se_w))
            output_mat[i][j] = max(filter_by_kernel(window, se))

    return output_mat

def dilation(matrix: Matrix, se: Matrix) -> Matrix:
    """
    Perform set theoretic dilation of a binary matrix using a specified structuring element.

    Parameters:
        `matrix` (Matrix): The original binary matrix where the dilation will be applied.
        `se` (Matrix): The structuring element that defines how the dilation is performed.

    Returns:
        Matrix: The resulting binary matrix after dilation.

    Description:
        This function expands the bright areas (1s) in the input matrix based on the structuring element.
        It first converts the structuring element into a set of relative positions.
        Then, for each bright pixel (1) in the input matrix,
        it adds the relative positions from the structuring element to create new bright pixels in the output matrix.
        This means that if a bright pixel is found, all positions defined by the structuring element around it will also become bright.
        The output matrix will have the same size as the input matrix.
    
    Examples:
    ```python
        >>> matrix = [[0, 1, 0],
        ...           [0, 0, 0],
        ...           [1, 0, 0]]
        >>> se = [[1, 1],
        ...       [1, 1]]
        >>> result = dilation(matrix, se)
        >>> result
        [[1, 1, 0]
        ,[1, 0, 0]
        ,[1, 0, 0]]
    """
    # Convert structuring element to a set of relative coordinates
    trans_set = structuring_element_to_set(se)
    foreground_pixels = pixel_coordinates(matrix)

    translated_set = set()
    # translate every pixel by the each value in translation set
    for pixel in foreground_pixels:
        for translation in trans_set:
            translated_set.add((pixel[0] + translation[0], pixel[1] + translation[1]))

    matrix_width = len(matrix[0])
    matrix_height = len(matrix)

    dilated_matrix = [[0 for _ in range(matrix_width)] for _ in range(matrix_height)]
    
    # Set the pixels in the result matrix to 1 where the dilated_pixels set contains coordinates
    for pixel_pos in translated_set:
        if 0 <= pixel_pos[0] < matrix_height and 0 <= pixel_pos[1] < matrix_width:
            dilated_matrix[pixel_pos[0]][pixel_pos[1]] = 1

    return dilated_matrix

def erosion(matrix: Matrix, se: Matrix) -> Matrix:
    """
    Perform erosion operation on a matrix using a given structuring element (se).

    Parameters:
        `matrix` (Matrix): The original binary matrix on which the erosion is applied.
        `se` (Matrix): The structuring element used for the erosion operation.

    Returns:
        Matrix: The result of the erosion operation on the matrix.

    Description:
        The erosion process involves sliding the `se` over the matrix and checking if all '1' pixels in the `se` align with '1' pixels in the matrix neighborhood.
        If they do, the center pixel of the neighborhood is set to '1' in the output matrix; otherwise, it is set to '0'.
        The output matrix has the same size as the input matrix.

    Examples:
    ```python
        >>> matrix = [[1, 1, 0, 0],
        ...           [1, 1, 0, 0],
        ...           [0, 0, 1, 1],
        ...           [0, 0, 1, 1]]
        >>> se = [[1, 1],
        ...       [1, 1]]
        >>> result = erosion(matrix, se)
        >>> result
        [[0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1]]
    """
    height = len(matrix)
    width = len(matrix[0])
    # se = _rotate_180_clockwise(se)
    se_h = len(se)
    se_w = len(se[0])

    # output will be same size as original in this case
    output_mat = [[0 for _ in range(width)] for _ in range(height)]

    # make bigger padded matrix 
    matrix = DN.pad(matrix,(se_w,se_h))

    for i in range(height):
        for j in range(width):
            window = slice_matrix(matrix,(i,i+se_h), (j,j+se_w))
            output_mat[i][j] = min(filter_by_kernel(window, se))

    return output_mat

def opening(matrix: Matrix, se: Matrix) -> Matrix:
    """
    Perform opening operation on a matrix using a given structuring element (se).

    Parameters:
        `matrix` (Matrix): The original binary matrix on which the opening operation is applied.
        `se` (Matrix): The structuring element used for the opening operation.

    Returns:
        Matrix: The result of the opening operation on the matrix.

    Description:
        The opening process consists of an erosion followed by a dilation using the same structuring element.
        This operation is useful for removing small objects from the foreground (usually taken as the bright pixels) of an image,
        placing them in the background, while preserving the shape and size of larger objects in the image.

    Examples:
        >>> matrix = [[0,0,0,0,0,0], 
        ...          [0,1,1,1,1,0], 
        ...          [0,1,1,0,1,0],
        ...          [0,1,1,0,1,0],
        ...          [0,1,1,1,1,0]]
        >>> se = [[1,1]
                ,[0,1]]
        >>> result = opening(matrix,se)
        >>> result
        [[0, 0, 0, 0, 0, 0]
        ,[0, 1, 1, 1, 1, 0]
        ,[0, 1, 1, 0, 1, 0]
        ,[0, 1, 1, 0, 0, 0]
        ,[0, 0, 1, 0, 0, 0]]
    """
    eroded = erosion(matrix, se)
    result = dilation(eroded, se)
    return result

def closing(matrix: Matrix, se: Matrix) -> Matrix:
    """
    Perform closing operation on a matrix using a given structuring element (se).

    Parameters:
        `matrix` (Matrix): The original binary matrix on which the closing operation is applied.
        `se` (Matrix): The structuring element used for the closing operation.

    Returns:
        Matrix: The result of the closing operation on the matrix.

    Description:
        The closing process consists of a dilation followed by an erosion using the same structuring element.
        This operation is useful for closing small holes in the foreground, bridging small gaps, and generally smoothing the outline of objects
        while keeping their sizes and shapes approximately the same.
    
    Examples:
        >>> matrix = [[0,0,0,0,0,0], 
        ...          [0,1,1,1,1,0], 
        ...          [0,1,1,0,1,0],
        ...          [0,1,1,0,1,0],
        ...          [0,1,1,1,1,0]]
        >>> se = [[1,1]
        ...     ,[0,1]]
        >>> result = closing(matrix,se)
        >>> result
        [[0, 0, 0, 0, 0, 0]
        ,[0, 1, 1, 1, 1, 0]
        ,[0, 1, 1, 1, 1, 0]
        ,[0, 1, 1, 1, 1, 0]
        ,[0, 1, 1, 1, 1, 0]]
    """
    dilated = dilation(matrix, se)
    result = erosion(dilated, se)
    return result

def internal_extraction(matrix: Matrix, se: Matrix) -> Matrix:
    """
    Extracts internal structures from a binary image by performing morphological erosion 
    followed by subtraction from the original image. This operation highlights inner 
    boundaries of objects in the image.

    Parameters:
        `matrix` (Matrix): Matrix Original binary image represented as a 2D list of 0s and 1s.
        `se` (Matrix): Structuring element used for erosion, represented as a 2D list.

    Returns:
        Matrix: Resulting binary image containing only the internal boundaries.

    Examples:
        >>> img = [[0, 0, 0, 0],
        ...        [0, 1, 1, 0],
        ...        [0, 1, 1, 0],
        ...        [0, 0, 0, 0]]
        >>> se = [[1, 1],
        ...       [1, 1]]
        >>> result = internal_extraction(img, se)
        >>> # Result will show the inner boundary of the 2x2 square
    """
    eroded_matrix = erosion(matrix, se)

    # Subtract the eroded image from the original image to get the external boundary
    result_matrix = [[DO.clip(matrix[i][j] - eroded_matrix[i][j],0,1) for j in range(len(matrix[0]))] for i in range(len(matrix))]
    
    return result_matrix

def external_extraction(matrix: Matrix, se: Matrix) -> Matrix:
    """
    Extracts external structures from a binary image by performing morphological dilation 
    followed by subtraction of the original image. This operation highlights outer 
    boundaries of objects in the image.

    Parameters:
        `matrix` (Matrix): Matrix Original binary image represented as a 2D list of 0s and 1s.
        `se` (Matrix): Structuring element used for erosion, represented as a 2D list.

    Returns:
        Matrix: Resulting binary image containing only the external boundaries.

    Examples:
        >>> img = [[0, 0, 0, 0],
        ...        [0, 1, 1, 0],
        ...        [0, 1, 1, 0],
        ...        [0, 0, 0, 0]]
        >>> se = [[1, 1],
        ...       [1, 1]]
        >>> result = external_extraction(img, se)
        >>> # Result will show the outer boundary surrounding the 2x2 square
    """

    dilated_matrix = dilation(matrix, se)
    
    # Subtract the original image from the dilated image to get the external boundary
    result_matrix = [[DO.clip(dilated_matrix[i][j] - matrix[i][j],0,1) for j in range(len(matrix[0]))] for i in range(len(matrix))]
    
    return result_matrix

def morphology_gradient(matrix: Matrix, se: Matrix) -> Matrix:
    """
    Computes the morphological gradient of a binary image by taking the difference 
    between dilated and eroded versions of the image. This operation highlights 
    all boundaries in the image.

    Parameters:
        `matrix` (Matrix): Matrix Original binary image represented as a 2D list of 0s and 1s.
        `se` (Matrix): Structuring element used for erosion, represented as a 2D list.

    Returns:
        Matrix: Resulting gradient image showing all boundaries.

    Examples:
        >>> img = [[0, 0, 0, 0],
        ...        [0, 1, 1, 0],
        ...        [0, 1, 1, 0],
        ...        [0, 0, 0, 0]]
        >>> se = [[1, 1],
        ...       [1, 1]]
        >>> result = morphology_gradient(img, se)
        >>> # Result will show both inner and outer boundaries
    """
    dilated_matrix = dilation(matrix, se)

    eroded_matrix = erosion(matrix, se)
    
    result_matrix = [[abs(dilated_matrix[i][j] - eroded_matrix[i][j]) for j in range(len(matrix[0]))] for i in range(len(matrix))]
    
    return result_matrix

def hole_filling(matrix: Matrix, se: Matrix, x_0_indx: Tuple[int,int],max_itiration:int = 100) -> Matrix:
    """
    Fills holes in binary images using iterative morphological reconstruction based 
    on a marker point. A hole is defined as a background region completely 
    surrounded by foreground pixels.

    Parameters:
        `matrix` (Matrix): binary image containing holes to be filled.
        `se` (Matrix): Structuring element.
        `x_0_indx` (Tuple[int,int]): Starting point coordinates (row, col) inside the hole to be filled. Must be a background pixel (0) surrounded by foreground pixels (1).
        `max_itiration` (int, optional): Maximum number of iterations for the filling process. Default is 100. Prevents infinite loops in case of non-convergence.

    Returns:
        Matrix: Binary image with the specified hole filled.

    Examples:
        >>> img = [[1, 1, 1, 1],
        ...        [1, 0, 0, 1],
        ...        [1, 0, 0, 1],
        ...        [1, 1, 1, 1]]
        >>> se = [[0, 1, 0],
        ...       [1, 1, 1],
        ...       [0, 1, 0]]
        >>> start_point = (1, 1)  # Coordinates inside the hole
        >>> filled_img = hole_filling(img, se, start_point)
        >>> # Result will fill the 2x2 hole with 1s
    """

    X_prev = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    X_prev[x_0_indx[0]][x_0_indx[1]] = 1
    complement_A = DO.complement(matrix,1)

    for _ in range(max_itiration):
        # Compute X_k = (X_{k-1} ⊕ B) ∩ A^c
        dilated_X_prev  = dilation(X_prev, se)
        X_k = intersection(dilated_X_prev, complement_A)

        # Check for convergence
        if X_k == X_prev:
            break

        # Update X_prev for next iteration
        X_prev = X_k

    # Combine filled hole with original foreground
    result = union(X_k, matrix)

    return result

def hole_filling_X_gen(matrix: Matrix, se: Matrix, x_0_indx: Tuple[int,int],max_itiration:int = 10) -> Generator[Matrix,None,None]:
    """
    Generator version of hole filling algorithm that yields intermediate results 
    during the filling process. Useful for visualizing the progression of the 
    hole filling operation.

    Parameters:
        `matrix` (Matrix): binary image containing holes to be filled.
        `se` (Matrix): Structuring element.
        `x_0_indx` (Tuple[int,int]): Starting point coordinates (row, col) inside the hole to be filled. Must be a background pixel (0) surrounded by foreground pixels (1).
        `max_itiration` (int, optional): Maximum number of iterations for the filling process. Default is 10. Lower than regular hole_filling as this is for visualization.

    Yields:
        Matrix: Intermediate binary images showing the progressive filling of the hole. Each yield represents one step in the iterative process.

    Examples:
        >>> img = [[1, 1, 1, 1],
        ...        [1, 0, 0, 1],
        ...        [1, 0, 0, 1],
        ...        [1, 1, 1, 1]]
        >>> se = [[0, 1, 0],
        ...       [1, 1, 1],
        ...       [0, 1, 0]]
        >>> start_point = (1, 1)
        >>> # get a list of 4 steps all filling
        >>> xs = [*hole_filling_X_gen(img, se, start_point, max_itiration = 4)]
        >>> plot_morphology(xs) # Custom function to show the matrix
    """
    X_prev = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    X_prev[x_0_indx[0]][x_0_indx[1]] = 1
    complement_A = DO.complement(matrix,1)
    yield X_prev

    for _ in range(max_itiration):
        # Compute X_k = (X_{k-1} ⊕ B) ∩ A^c
        dilated_X_prev  = dilation(X_prev, se)
        X_k = intersection(dilated_X_prev, complement_A)

        # Check for convergence
        if X_k == X_prev:
            yield X_k
            break

        # Update X_prev for next iteration
        X_prev = X_k
        yield X_k

def hit_or_miss(matrix: Matrix, se1: Matrix, se2: Matrix) -> Matrix:
    """
    Performs hit-or-miss transform (HMT) to detect specific patterns or templates 
    in binary images using two complementary structuring elements.

    Parameters:
        `matrix` (Matrix): binary image containing holes to be filled.
        `se1` (Matrix): First structuring element that matches foreground pixels (1s).
        `se2` (Matrix): Second structuring element that matches background pixels (0s).

    Returns:
        Matrix: Binary image where 1s indicate locations where the pattern was found. Matches occur where both structuring elements align with their respective foreground and background requirements.

    Notes:
        1. The hit-or-miss transform is particularly useful for:
            - Template matching in binary images
            - Corner detection
            - Endpoint detection
            - Specific shape detection

    Examples:
        >>> # Examples: Detecting upper-left corners
        >>> img = [[1, 1, 0],
        ...        [1, 1, 0],
        ...        [0, 0, 0]]
        >>> # SE1 for required foreground pixels
        >>> se1 = [[1, 1],
        ...        [1, 0]]
        >>> # SE2 for required background pixels
        >>> se2 = [[0, 0],
        ...        [0, 1]]
        >>> result = hit_or_miss(img, se1, se2)
        >>> # Result will mark the upper-left corner location
    """
    complement_A = DO.complement(matrix,1)

    A_erosion = erosion(matrix,se1)
    A_c_dilation = erosion(complement_A,se2)
    result = intersection(A_erosion,A_c_dilation)
    return result
