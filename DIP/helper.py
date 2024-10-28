from typing import List, Optional, Tuple, Union, Set

import math as m
from random import randint

Matrix = List[List[int]]

def flatten_matrix(matrix: Matrix)->Matrix:
    """
    Flattens a 2D matrix into a 1D list.

    Parameters:
        `matrix` (Matrix): A 2D list of integers.

    Returns:
        Matrix: A 1D list containing all elements from the input matrix.

    Examples:
        >>> flatten_matrix([[1, 2], [3, 4]])
        [1, 2, 3, 4]
    """
    return [item for row in matrix for item in row]

def slice_matrix(matrix: Matrix, row_slice: Tuple[int, int], col_slice: Tuple[int, int]) -> Matrix:
    """
    Slices a matrix based on specified row and column indices.

    Parameters:
        `matrix` (Matrix): The input matrix to slice.
        `row_slice` (Tuple[int, int]): A tuple indicating the start and end indices for rows.
        `col_slice` (Tuple[int, int]): A tuple indicating the start and end indices for columns.

    Returns:
        Matrix: The sliced matrix.

    Examples:
        >>> mat =  [[1, 2, 3],
                    [4, 5, 6]]
        >>> slice_matrix(mat, (0, 2), (1, 3))
        [[2, 3], [5, 6]]
    """
    row_start, row_end = row_slice
    col_start, col_end = col_slice
    # Slicing beyond the length of the matrix does not throw an error
    return [row[col_start:col_end] for row in matrix[row_start:row_end]]

def filter_by_kernel(window: Matrix, kernel: Matrix, equal_to:int = 1) -> List[int]:
    """
    Filters values from a window matrix based on a kernel matrix.

    Parameters:
        `window` (Matrix): The input window matrix.
        `kernel` (Matrix): The kernel matrix used for filtering.
        `equal_to` (int): The value to filter by. Defaults to 1.

    Returns:
        List[int]: A list of values from the window that correspond to the kernel's specified value.

    Examples:
        >>> mat = [[1, 2], [3, 4]] 
        >>> ker = [[0, 1], [1, 0]]
        >>> filter_by_kernel(mat, ker)
        [2, 3]
    """
    return [value for value, k in zip(flatten_matrix(window), flatten_matrix(kernel)) if k == equal_to]

def structuring_element_to_set(structuring_element: Matrix, center_pos: Tuple[int,int] = None) -> Set[Tuple[int,int]]:
    """
    Converts a structuring element into a set of relative coordinates based on its center position.

    Parameters:
        `structuring_element` (Matrix): The structuring element as a binary matrix.
        `center_pos` (Tuple[int,int] | None): The center position. If None, the center is calculated.

    Returns:
        Set[Tuple[int,int]]: A set of relative coordinates where the structuring element is equal to 1.

    Examples:
        >>> convert_structuring_element_to_set([[0, 1], [1, 0]])
        {(-1, 0), (0, -1)}
    """
    translation_set = set()

    kernel_width = len(structuring_element[0])
    kernel_height = len(structuring_element)
    
    if center_pos == None:
        center_pos = get_center(kernel_width,kernel_height)
    
    # Convert the structuring element to a set of relative coordinates
    for row in range(kernel_height):
        for col in range(kernel_width):
            if structuring_element[row][col] == 1:
                translation_set.add((row - center_pos[0], col - center_pos[1]))
    return translation_set

def pixel_coordinates(image: Matrix, equal_to: int = 1) -> Set[Tuple[int,int]]:
    """
    Returns a set of coordinates where the pixel value equals the specified integer.

    Parameters:
        `image` (Matrix): The input image as a matrix.
        `equal_to` (int): The pixel value to search for. Defaults to 1.

    Returns:
        Set[Tuple[int,int]]: A set of tuples representing the coordinates of matching pixels.

    Examples:
        >>> pixel_coordinates([[0, 2], [1, 2], [1, 3]], equal_to=2)
        {(0, 1), (1, 1)}
    """
    coordinates = set()
    for r in range(len(image)):
        for c in range(len(image[0])):
            if image[r][c] == equal_to:
                coordinates.add((r,c))
    return coordinates

def make_random(width: int = 3, height: int = 3, bit_depth: int = 8) -> Matrix:
    """
    Create a random matrix with specified dimensions and bit depth.

    Parameters:
        `width` (int): The number of columns in the matrix. Defaults to 3.
        `height` (int): The number of rows in the matrix. Defaults to 3.
        `bit_depth` (int): The bit depth for the random values. Defaults to 8.

    Returns:
        `Matrix`: A matrix of the specified dimensions filled with random integers based on the bit depth.

    Examples:
        >>> make_random(2, 2)
        [[23, 45], [12, 67]] # Output will vary
    """

    max_value = 2 ** bit_depth
    matrix = [[randint(0, max_value - 1) for _ in range(width)] for _ in range(height)]
    return matrix

def get_center(width: int = 3, height: int = 3, return_matrix: bool = False) -> Union[Matrix, Tuple[int, int]]:
    """
    Get the center coordinate of a matrix of the specified shape or display it.

    Parameters:
        `width` (int): The number of columns in the matrix. Defaults to 3.
        `height` (int): The number of rows in the matrix. Defaults to 3.
        `return_matrix` (bool): If True returns a marked center coordinate in a visual format. Defaults to False.

    Returns:
        If `return_matrix` is True returns a visual representation with "X" at center.
        If False returns (x,y) indices of the center coordinate.

    Examples:
        >>> get_center(5, 5)
        (2, 2)

        >>> matrix = get_center(5, 5, return_matrix=True)
        >>> display_matrices([matrix], coordinates=True)
            0 1 2 3 4
        0 [ * * * * * ]
        1 [ * * * * * ]
        2 [ * * X * * ]
        3 [ * * * * * ]
        4 [ * * * * * ]
    """
    x,y = height//2,width//2

    if return_matrix:
        return [["*" if row != x or col != y else "X" for col in range(width)] for row in range(height)]
    else:
        return x,y

def get_pixel(matrix: Matrix, index: Tuple[int, int]) -> int:
    """
    Retrieve the value of a specific pixel in a matrix.

    Parameters:
        `matrix` (Matrix): The input matrix.
        `index` (Tuple[int,int]): The coordinates of the pixel to retrieve.

    Returns:
        int: The value at the specified pixel location.

    Examples:
        >>> matrix = [[1, 2, 3],
        ...           [4, 5, 6],
        ...           [7, 8, 9]]
        >>> get_pixel(matrix, (1, 2))
        6
    """
    return matrix[index[0]][index[1]]

def display_matrices(list_of_matrices: List[Matrix], text: Optional[List[str]] = None, coordinates: bool = False) -> None:
    """
    Display a list of matrices with optional coordinate labels.
    Each matrix is displayed with elements right-aligned for better readability.

    Parameters:
        `list_of_matrices` (List[Matrix]): A list of matrices to be displayed.
        `text` (List[str], Optional): Optional text labels for each matrix.
        `coordinates` (bool): If True display row and column numbers. Defaults to False.

    Raises:
        ValueError: If the length of text doesn't match the number of matrices

    Examples:
        >>> list_of_matrices = [
        ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ...     [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
        ... ]
        >>> display_matrices(list_of_matrices, coordinates=True)
        Matrix 1:
            0 1 2
        0 [ 1 2 3 ]
        1 [ 4 5 6 ]
        2 [ 7 8 9 ]

        Matrix 2:
            0  1  2
        0 [ 10 20 30 ]
        1 [ 40 50 60 ]
        2 [ 70 80 90 ]
    """
    # Input validation
    if not list_of_matrices:
        return
    
    # Handle text labels
    if text is None:
        if len(list_of_matrices) == 1:
            labels = [""]  # Single matrix: no label
        else:
            labels = [f"Matrix {i + 1}:" for i in range(len(list_of_matrices))]  # Multiple matrices: numbered
    else:
        if len(text) != len(list_of_matrices):
            raise ValueError("The length of list_of_matrices and text must be the same")
        labels = text

    # Process each matrix
    for matrix, label in zip(list_of_matrices, labels):
        # Print the matrix label if it's not empty
        if label:
            print(label)
        
        # Calculate the maximum length of any element in the matrix for uniform column width
        max_length = max(len(str(item)) for row in matrix for item in row)

        # Print column numbers as headers if coordinates is True
        if coordinates:
            print("    ", end="")  # Initial spacing for alignment
            for col_num in range(len(matrix[0])):
                print(str(col_num).rjust(max_length), end=" ")  # Right-align each column number
            print()  # Newline after headers

        # Print each row of the matrix with appropriate spacing and brackets
        for idx, row in enumerate(matrix):
            # Print row number if coordinates is True
            if coordinates:
                print(f"{idx}", end=" ")

            # Format and print the row
            formatted_row = " ".join(
                str(val).rjust(max_length) 
                for val in row
            )
            print(f"[ {formatted_row} ]")
        # Print an empty line to separate matrices
        print()

def normal_round(float_to_round: float) -> int:
    """
    Round a floating-point number to the nearest integer (not round half to even).

    Parameters:
        float_to_round (float): The number to round.

    Returns:
        int: The rounded integer value.

    Examples:
        >>> normal_round(2.3)
        2
        >>> normal_round(2.5)
        3
        >>> normal_round(2.8)
        3
    """

    return int(float_to_round + 0.5)

def clip(x:int,min_value:int, max_value:int)->int:
    """
    Clips an integer value between min_value and max_value.

    Parameters:
        x (int): The integer value to clip.
        min_value (int): The minimum allowed value.
        max_value (int): The maximum allowed value.

    Returns:
        int: The clipped value.

    Examples:
        >>> clip(10,5 ,15)
        10
        >>> clip(4 ,5 ,15)
        5
        >>> clip(20 ,5 ,15)
        15
    """
    return min(max(x, min_value), max_value)

def mat_max(matrix: Matrix) -> int:
    """
    Finds the maximum value in a given matrix.

    Parameters:
        matrix (Matrix): A list of lists representing the input matrix.

    Returns:
        int: The maximum value found.

    Examples:
        >>> mat_max([[1 ,2 ,3],[4 ,5 ,6]])
        6
    """

    return max(max(row) for row in matrix)

def intersection(matrix1: Matrix, matrix2: Matrix) -> Matrix:
    """
    Computes the pixel-wise intersection of two grayscale or binary images by
    taking the minimum value of corresponding pixels in each matrix.

    For binary images, this operation is equivalent to a logical AND, where 
    only overlapping regions with value 1 in both images will retain a value of 1. 
    For grayscale images, it selects the darker intensity for each pixel.

    Parameters:
        `matrix1` (Matrix): First input matrix.
        `matrix2` (Matrix): Second input matrix, with the same dimensions as matrix1.

    Returns:
        Matrix: A new matrix containing minimum values from both matrices.

    Examples:
        >>> m1 = [[1, 0], [1, 1]]
        >>> m2 = [[1, 1], [0, 1]]
        >>> intersection(m1, m2)
        [[1, 0], [0, 1]]
    """
    result = [[min(matrix1[i][j], matrix2[i][j]) for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
    return result

def union(matrix1: Matrix, matrix2: Matrix) -> Matrix:
    """
    Computes the pixel-wise union of two binary or grayscale images by
    taking the maximum value of corresponding pixels in each matrix.

    For binary images, this operation is equivalent to a logical OR, where 
    a pixel in the result is 1 if at least one of the corresponding pixels 
    in matrix1 or matrix2 is 1. For grayscale images, it selects the brighter 
    intensity for each pixel.

    Parameters:
        `matrix1` (Matrix): The first image matrix.
        `matrix2` (Matrix): The second image matrix, with the same dimensions as matrix1.

    Returns:
        Matrix: A new matrix containing maximum values from both matrices.

    Examples:
        >>> m1 = [[1, 0], [1, 1]]
        >>> m2 = [[1, 1], [0, 1]]
        >>> union(m1, m2)
        [[1, 1], [1, 1]]
    """
    result = [[max(matrix1[i][j], matrix2[i][j]) for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
    return result

def bit_depth_from_mat(matrix: Matrix)->int:
    """
    Calculates bit depth required to represent values in given matrix.

    Parameters:
        `matrix` (Matrix): Input data as a list of lists.

    Returns:
        int: Bit depth calculated using logarithm base two.

    Examples:
        >>> bit_depth_from_mat([[12,120],[15,80]])  
        7
    """
    return m.ceil(m.log2(mat_max(matrix) + 1))

def resize(matrix: Matrix, size: Tuple[int, int]) -> Matrix:
    """
    This function resizes a given matrix `matrix` to the specified dimensions `size` using a custom algorithm. It spreads the original values from the source matrix into the resized matrix, placing empty cells in between the original values. It then fills in the empty cells based on the values of their neighboring cells.
    
    Steps:
        1. Calculate the number of empty cells between the original values.
        2. Create a new resized matrix with the specified dimensions, initially filled with zeros.
        3. Spread the original values into the new matrix at appropriate intervals.
        4. Fill in the empty cells:
            - For empty row cells, take the average of the cells above and below.
            - For empty column cells, take the average of the cells to the left and right.
            - For empty diagonal cells, take the average of the four neighboring diagonal cells.

    Parameters:
        `matrix` (Matrix): The source matrix to be resized.
        `size` (Tuple[int, int]): A tuple specifying the dimensions (`dst_rows`, `dst_cols`) of the resized matrix.

    Returns:
        Matrix: The resized matrix.

    Examples:
        >>> matrix = [
        ...     [1, 2, 3],
        ...     [4, 5, 6],
        ...     [7, 8, 9]
        ... ]
        >>> size = (6, 6)
        >>> resized_matrix = resize(matrix, size)
        >>> for row in resized_matrix:
        ...     print(row)
        [1, 1, 2, 2, 3, 3]
        [2, 2, 3, 3, 4, 4]
        [4, 4, 5, 5, 6, 6]
        [5, 5, 6, 6, 7, 7]
        [7, 7, 8, 8, 9, 9]
        [8, 8, 9, 9, 10, 10]
    """
    src_rows, src_cols = len(matrix), len(matrix[0])
    dst_rows, dst_cols = size

    # Calculate the number of empty cells between original values
    row_spacing = (dst_rows - src_rows) // (src_rows - 1) if src_rows > 1 else 0
    col_spacing = (dst_cols - src_cols) // (src_cols - 1) if src_cols > 1 else 0

    # Create a new resized matrix with empty cells
    resized_mat = [[0 for _ in range(dst_cols)] for _ in range(dst_rows)]

    if row_spacing < 0 or col_spacing < 0:
        return resized_mat

    # Spread original values into the resized matrix with empty cells
    for i in range(src_rows):
        for j in range(src_cols):
            resized_mat[(row_spacing + 1) * i][(col_spacing + 1) * j] = matrix[i][j]

    # Fill in the empty cells
    for i in range(dst_rows):
        for j in range(dst_cols):
            # For empty row cells
            if i % (row_spacing + 1) != 0:
                # Pixel value = round of average of right and left cells if not at last row, if at it copy last row
                resized_mat[i][j] = normal_round((resized_mat[i-1][j] + resized_mat[min(i+1, dst_rows-1)][j]) / 2) if i != dst_rows - 1 else resized_mat[i-1][j]
            # For empty column cells
            elif j % (col_spacing + 1) != 0:
                resized_mat[i][j] = normal_round((resized_mat[i][j-1] + resized_mat[i][min(j+1, dst_cols-1)]) / 2) if j != dst_cols - 1 else resized_mat[i][j-1]
            # For empty diagonal cells
            elif i % (row_spacing + 1) != 0 and j % (col_spacing + 1) != 0:
                sum_neighbors = 0
                count_neighbors = 0
                if i > 0 and j > 0:
                    sum_neighbors += resized_mat[i-1][j-1]
                    count_neighbors += 1
                if i > 0 and j < dst_cols - 1:
                    sum_neighbors += resized_mat[i-1][j+1]
                    count_neighbors += 1
                if i < dst_rows - 1 and j > 0:
                    sum_neighbors += resized_mat[i+1][j-1]
                    count_neighbors += 1
                if i < dst_rows - 1 and j < dst_cols - 1:
                    sum_neighbors += resized_mat[i+1][j+1]
                    count_neighbors += 1
                resized_mat[i][j] = sum_neighbors // count_neighbors if count_neighbors > 0 else 0

    return resized_mat

def rotate_90(matrix: Matrix, clockwise: bool = True) -> Matrix:
    """
    Rotates a matrix 90 degrees clockwise (default) or counterclockwise.

    Parameters:
        `matrix` (Matrix): The matrix to rotate.
        `clockwise` (bool): If True (default), rotates clockwise; if False, rotates counterclockwise.

    Returns:
        rotated(Matrix): The rotated matrix.
    Examples:
        >>> rotate_90([[10 ,20],
        ...            [30 ,40]])
        [[30 ,10]
        ,[40 ,20]]
    """


    if not matrix:
        return []

    rows = len(matrix)
    cols = len(matrix[0])

    # Transpose the matrix
    transposed = [[matrix[j][i] for j in range(rows)] for i in range(cols)]

    if clockwise:
        # Reverse each row to get the clockwise rotation
        return [list(reversed(row)) for row in transposed]
    else:
        # Reverse the order of rows to get the counterclockwise rotation
        return transposed[::-1]

def _rotate_180(matrix: Matrix) -> Matrix:
    """Rotates a matrix 180 degrees."""
    if not matrix: # if None
        return []

    # Reverse each row and each column
    return [row[::-1] for row in matrix[::-1]]

def make_realistic_matrix(
    width: int = 9, 
    height: int = 9, 
    bit_depth: int = 1, 
    elements_num: int = 2,
    background_intensity_factor: float = 0.2,
    foreground_intensity_factor: float = 0.5,
    size_factor: float = 0.4  # Controls the size of objects (0.0 to 1.0)
) -> List[List[int]]:
    """
    Generate a realistic matrix with background and foreground objects.
    
    Parameters:
        width: Width of the matrix
        height: Height of the matrix
        bit_depth: Bit depth for intensity values
        elements_num: Number of foreground objects to generate
        background_intensity_factor: Max background intensity as fraction of max value
        foreground_intensity_factor: Min foreground intensity as fraction of max value
        size_factor: Controls size of objects (0.0 to 1.0)
    
    Returns:
        Matrix with realistic intensity distribution
    Examples:
    >>> helper.make_realistic_matrix()
    ... [ 0 0 0 0 0 0 0 0 0 ]
    ... [ 0 0 0 0 0 0 0 0 0 ]
    ... [ 0 0 0 0 0 0 0 0 0 ]
    ... [ 0 0 0 0 0 1 1 1 0 ]
    ... [ 0 0 0 0 0 1 1 1 0 ]
    ... [ 0 0 0 0 0 1 1 1 0 ]
    ... [ 0 0 0 1 1 1 1 1 0 ]
    ... [ 0 0 0 1 1 1 1 0 0 ]
    ... [ 0 0 0 1 1 1 1 0 0 ]
    """
    from DIP.neighborhood_operations import Kernels, apply_filter
    
    # Input validation
    if width < 1 or height < 1:
        raise ValueError("Width and height must be positive")
    if bit_depth < 1 or bit_depth > 32:
        raise ValueError("Bit depth must be between 1 and 32")
    if elements_num < 0:
        raise ValueError("Number of elements must be non-negative")
    if not 0.0 <= size_factor <= 1.0:
        raise ValueError("Size factor must be between 0.0 and 1.0")
        
    max_value = 2**bit_depth - 1
    
    # Define intensity ranges
    background_max = int(max_value * background_intensity_factor)
    foreground_min = int(max_value * foreground_intensity_factor)
    
    # Create background
    matrix = [[randint(0, background_max) for _ in range(width + 2)] 
             for _ in range(height + 2)]
    
    # Generate foreground objects
    for _ in range(elements_num):
        height, width = len(matrix), len(matrix[0])
        
        # Calculate object size based on size_factor
        obj_size = max(1, int(min(width, height) * size_factor))
        obj_w = randint(int(obj_size * .75),obj_size)
        obj_h = randint(int(obj_size * .75),obj_size)
        
        # Generate random position ensuring object fits within matrix
        obj_x = randint(0, height - obj_size)
        obj_y = randint(0, width - obj_size)
        
        # Add square object to matrix
        for i in range(obj_x, obj_x + obj_h):
            for j in range(obj_y, obj_y + obj_w):
                if i < height and j < width:  # Boundary check
                    if bit_depth == 1:
                        matrix[i][j] = 1
                    else:
                        matrix[i][j] = randint(foreground_min, max_value)
    
    matrix = apply_filter(matrix, Kernels.weighted_smoothing_filter, bit_depth=bit_depth)
    
    return matrix

def max_dimensions(matrices: List[Matrix]) -> Tuple[int, int]:
    """
    Calculates the maximum width and height from a list of 2D matrices.

    Parameters:
        matrices (List[Matrix]]): A list of 2D matrices represented as lists of lists.

    Returns:
        Tuple[int, int]: A tuple containing the maximum width and height:
                        (max_width, max_height), where
                        - max_width is the maximum number of columns in any matrix,
                        - max_height is the maximum number of rows in any matrix.

    Examples:
    >>> matrices = [
    ...     [[1, 0], [0, 1]],
    ...     [[1, 1, 1], [0, 0, 0]],
    ...     [[1]]
    ... ]
    >>> max_dimensions(matrices)
    (3, 2)
    """
    max_width = max(len(matrix[0]) for matrix in matrices if matrix)
    max_height = max(len(matrix) for matrix in matrices if matrix)
    return max_width, max_height

from typing import List, Tuple

def trim(matrix: Matrix, trim_amount: Tuple[int, int]) -> Matrix:
    """
    Trims a specified number of rows and columns from the edges of a matrix.

    Parameters:
        matrix (Matrix): The original 2D matrix to be trimmed.
        trim_amount (Tuple[int, int]): A tuple (width, height) where:
                                        - width is the number of columns to remove from each side,
                                        - height is the number of rows to remove from each side.

    Returns:
        Matrix: The trimmed matrix with dimensions reduced by the specified width and height.
                        If the trim amount is greater than half the matrix dimensions, an empty matrix is returned.

    Example:
    >>> matrix = [
    ...     [1, 2, 3, 4],
    ...     [5, 6, 7, 8],
    ...     [9, 10, 11, 12],
    ...     [13, 14, 15, 16]
    ... ]
    >>> trim(matrix, (1, 1))
    [[6, 7], [10, 11]]
    """
    trim_width, trim_height = trim_amount
    num_rows = len(matrix)
    num_cols = len(matrix[0]) if num_rows > 0 else 0

    # Ensure trimming does not exceed half the matrix size
    if trim_width * 2 >= num_cols or trim_height * 2 >= num_rows:
        return []  # Return empty matrix if trimming exceeds matrix dimensions

    # Trim rows and columns from each side
    trimmed_matrix = matrix[trim_height:num_rows - trim_height]
    trimmed_matrix = [row[trim_width:num_cols - trim_width] for row in trimmed_matrix]

    return trimmed_matrix
