from copy import deepcopy
from typing import Generator, List, Optional,Tuple
import math as m
from DIP.helper import Matrix, get_center, rotate_90, slice_matrix, bit_depth_from_mat, clip, flatten_matrix
import DIP.operations as DP


class Kernels:
    """
    Collection of common image processing kernels used for filtering and edge detection.
    
    Each kernel is represented as a 2D list of coefficients that can be applied to an image
    through convolution or correlation operations.
    """

    # Averaging filter with uniform weights
    smoothing_filter = [
        [0.111, 0.111, 0.111],
        [0.111, 0.111, 0.111],
        [0.111, 0.111, 0.111]
    ]  # 1/9 weight for each pixel

    # Gaussian-like smoothing filter with center-weighted coefficients
    weighted_smoothing_filter = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625]
    ]  # Approximates Gaussian blur

    # Enhancement filter that emphasizes edges
    high_pass_filter = [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ]

    # Roberts cross-gradient operators
    robert_operator = [
        [-1, 0],
        [0, 1]
    ]  # Detects diagonal edges

    # Prewitt gradient operators
    prewitt_operator = [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]  # Detects vertical edges (horizontal gradient)

    # Sobel gradient operators
    sobel_operator = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]  # Enhanced edge detection with central emphasis

    # Laplacian operator for edge detection
    laplacian_operator = [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ]  # Second derivative operator

    # 3*3 gaussian kernel with sigma = 2
    gaussian_kernel = [
        [0.1019, 0.1154, 0.1019],
        [0.1154, 0.1308, 0.1154],
        [0.1019, 0.1154, 0.1019]
        ]

def pad(matrix: Matrix, kernel_shape: Tuple[int,int],pad_with: int = 0) -> Matrix:
    """
    Pads a matrix with a specified value.

    Parameters:
        `matrix` (Matrix): The original matrix to be padded.
        `kernel_shape` (Tuple[int,int]): (width, height) of the kernel, used to determine the padding.
        `pad_with` (int): Value to pad the matrix with. Defaults is 0.

    Returns:
        Matrix: The padded matrix.
    
    Raises:
        ValueError: If pad_value is not an integer
    
    Examples:
    >>> mat=[[3,1,5],
            [3,0,1],
            [5,4,1],
            [0,2,2]]
    >>> display_matrices([pad(mat, (2,4))])
    ... [ 0 0 0 0 ]
    ... [ 0 0 0 0 ]
    ... [ 0 3 1 5 ]
    ... [ 0 3 0 1 ]
    ... [ 0 5 4 1 ]
    ... [ 0 0 2 2 ]
    ... [ 0 0 0 0 ]
    """

    if not isinstance(pad_with, int):
        raise ValueError("pad_with must be an integer")
    width = len(matrix[0])

    k_w, k_h = kernel_shape
    # get the how much center is away from each corner of the kernel
    # the index is number of rows or column away from this point
    pad_top,pad_left = get_center(k_w,k_h)
    # lenght - indx = lenght - 1 = indx
    pad_right = k_w - pad_left - 1
    pad_bottom = k_h - pad_top - 1

    padded_width = width + pad_left + pad_right
    
    # Create top padding rows
    result = [[pad_with] * padded_width for _ in range(pad_top)]
    
    # Add rows with content
    for row in matrix:
        padded_row = [pad_with] * pad_left + row + [pad_with] * pad_right
        result.append(padded_row)
    
    # Add bottom padding rows
    result.extend([[pad_with] * padded_width for _ in range(pad_bottom)])
    
    return result

def get_pixel_neighborhoods(matrix: Matrix, kernel_dim: Tuple[int, int], pixel_pos: Tuple[int, int]) -> Matrix:
    """
    Extracts a neighborhood of a pixel from a matrix based on the given kernel size.

    Parameters:
        `matrix` (Matrix): The original matrix from which to extract the neighborhood.
        `pixel_pos` (Tuple[int, int]): The position of the pixel around which to extract the neighborhood.
        `kernel_dim` (Tuple[int, int]): Dimensions of the kernel used to define the neighborhood size.

    Returns:
        Matrix: A matrix representing the neighborhood of the specified pixel.

    Examples:
        >>> matrix = [[1, 2, 3, 4],
        ...           [5, 6, 7, 8],
        ...           [9, 10, 11, 12],
        ...           [13, 14, 15, 16]]
        >>> pixel_pos = (1, 1)
        >>> kernel_dim = (3, 3)
        >>> neighborhood = neighborhoods_of_pixel(matrix, pixel_pos, kernel_dim)
        >>> neighborhood
        [[1, 2, 3],
        [5, 6, 7],
        [9, 10, 11]]
    """


    width = len(matrix[0])
    height = len(matrix)

    k_w = kernel_dim[0]
    k_h = kernel_dim[1]
    left,top = get_center(k_w,k_h)
    # lenght - indx = lenght - 1 = indx
    right = k_w - left - 1
    down = k_h - top - 1

    # Calculate the starting position for the neighborhood
    start_row = max(0, pixel_pos[0] - left)
    start_col = max(0, pixel_pos[1] - top)

    # Calculate the ending position for the neighborhood
    end_row = min(height - 1, pixel_pos[0] + down)
    end_col = min(width - 1, pixel_pos[1] + right)

    # Calculate the size of the neighborhood matrix
    neighborhood_width = end_col - start_col + 1
    neighborhood_height = end_row - start_row + 1

    # Create a zero-filled matrix of the shape of the kernel
    output_matrix = [[0 for _ in range(k_w)] for _ in range(k_h)]

    # Calculate the shift amount
    shift_row = max(0, top - pixel_pos[0])
    shift_col = max(0, left - pixel_pos[1])

    # Copy values from the original matrix to the output matrix
    for i in range(neighborhood_height):
        for j in range(neighborhood_width):
            output_matrix[i + shift_row][j + shift_col] = matrix[start_row + i][start_col + j]

    return output_matrix

def _get_neighborhood_and_center(matrix: Matrix, kernel_dim: Tuple[int,int], pad_with:int|None = None) -> Generator[Matrix,None,None]:
    """
    Generate neighborhoods of a matrix based on a given kernel, including the center pixel coordinates.

    Parameters:
        `matrix` (Matrix): The original matrix from which neighborhoods are extracted.
        `kernel_dim` (Tuple[int,int]): (`width`, `height`) The kernel dimensions used to determine the size and shape of neighborhoods.

    Yields:
        `Tuple[Matrix,Tuple[int,int]]`: A tuple containing the neighborhood matrix and the coordinates of its center pixel.

    Description:
        This generator yields tuples where the first element is the neighborhood matrix and the second element is a tuple containing the coordinates of the center pixel of the neighborhood.

    Examples:
    ```python
    >>> matrix = [[1, 2],
    ...           [3, 4]]
    >>> kernel_dim = (2, 2)
    >>> gen = _neighborhood_with_center(matrix, kernel_dim)
    >>> for neighborhood, center in gen:
    ...     print("Neighborhood:", neighborhood)
    ...     print("Center coordinates:", center)
    Neighborhood: [[1, 2], [3, 4]]
    Center coordinates: (1, 1)
    """

    k_w = kernel_dim[0]
    k_h = kernel_dim[1]
    k_center = get_center(k_w,k_h)

    # if the matrix is padded we start from the original matrix start
    offset_w,offset_h = 0,0

    if pad_with is None:
        width = len(matrix[0])
        height = len(matrix)
    elif isinstance(pad_with,int):
        matrix = pad(matrix,(k_w,k_h),pad_with)
        width = len(matrix[0])
        height = len(matrix)
        # subtract the added offset to get the original matrix
        offset_w,offset_h = get_center(k_w,k_h)
    else:
        raise ValueError("padding must be None or integer")
    
    for row in range(height - k_h + 1):
        for col in range(width - k_w + 1):
            # getting the values same size as the filter 
            neighbors_with_center = slice_matrix(matrix,(row,row + k_h),(col, col + k_w)),(row+k_center[0] - offset_h,col + k_center[1] - offset_w)
            yield neighbors_with_center

def correlate(matrix: Matrix, kernel: Matrix, pad_with:Optional[int] = None, bit_depth:Optional[int] = None) -> Matrix:
    """
    Perform convolution operation on a matrix using a given kernel.

    Parameters:
        `matrix` (Matrix): The original matrix on which convolution is applied.
        `kernel` (Matrix): The convolution kernel.
        `pad_with` (int, optional): Padding value border handling. If `None`, no padding is applied, output is smaller than input. Defaults to `None`.

    Returns:
        `Matrix`: The result of convolving the matrix with the kernel.

    Description:
        This function performs correlation operation on the `matrix` using the specified `kernel`.
        It supports optional padding with a specified value.
        Correlation is performed by sliding the kernel over the matrix and computing the dot product between the kernel and the neighborhood of each pixel.
    Examples:
    >>> matrix = [[1, 2, 3],
    ...           [4, 5, 6],
    ...           [7, 8, 9]]
    >>> kernel = [[1, 0, -1],
    ...           [1, 0, -1],
    ...           [1, 0, -1]]
    >>> result = correlate(matrix, kernel,pad_with=0)
    >>> result
    [[-7, -4, 7],
    [-15, -6, 15],
    [-13, -4, 13]]
    """
    if bit_depth is None:
        bit_depth = bit_depth_from_mat(matrix)

    max_value = 2 ** bit_depth - 1
    
    height = len(matrix)
    width = len(matrix[0])
    k_w = len(kernel[0])
    k_h = len(kernel)

    if pad_with is None:
        # make smaller matrix
        output_mat = [[pad_with for _ in range(width - k_w + 1)] for _ in range(height - k_h + 1)]

    elif isinstance(pad_with,int):
        # output will be same size as original in this case
        output_mat = [[pad_with for _ in range(width)] for _ in range(height)]

        matrix = pad(matrix,(k_w,k_h),pad_with)
        # get the new dimentions to work on
        height = len(matrix)
        width = len(matrix[0])
    else:
        raise ValueError("padding must be None or integer")

    for row in range(height - k_h + 1):
        for col in range(width - k_w + 1):
            # getting the values same size as the filter 
            neighborhood = slice_matrix(matrix,(row,row + k_h),(col, col + k_w))
            # for every row in neighborhood and kernel and for each element in both rows multiply elements then sum all
            output_mat[row][col] = clip(round(sum(e * v for e,v in zip(flatten_matrix(neighborhood), flatten_matrix(kernel))),2),0,max_value)

    return output_mat

def rank_order_filter(matrix, kernel_dim:Tuple[int,int], rank:int, pad_with:int|None = None) -> Matrix:
    """
    Apply rank-order filter on a matrix using a given kernel size and rank.

    Parameters:
        `matrix` (Matrix): The original matrix on which the filter is applied.
        `kernel_dim` (Tuple[int,int]): (`width`, `height`) The dimensions of the filter kernel.
        `rank` (int): The rank used in the rank-order filter, starts from 1 to the size of the kernel (width * height).
        `pad_with` (int, optional): The value used for padding the matrix. If `None`, no padding is applied. Defaults to `None`.

    Returns:
        Matrix: The result of applying the rank-order filter on the matrix.

    Description:
        This function applies a rank-order filter on the `matrix` using a filter kernel of specified dimensions (`kernel_dim`).
        The `rank` parameter determines which element in the sorted neighborhood values is selected as the output pixel value.

    Examples:
    >>> matrix = [[1, 2, 3],
    ...           [4, 5, 6],
    ...           [7, 8, 9]]
    >>> kernel_dim = (3, 3)
    >>> rank = 3
    >>> result = rank_order_filter(matrix, kernel_dim, rank, pad_with = 0)
    >>> result
    [[0, 0, 0]
    ,[0, 3, 0]
    ,[0, 0, 0]]
    """

    k_width = kernel_dim[0]
    k_hieght = kernel_dim[1]
    if rank > k_width * k_hieght:
        raise ValueError("rank must be less than or equal the number of elements in the filter")

    output_mat = deepcopy(matrix)

    for neighbors, center in _get_neighborhood_and_center(matrix,(k_width,k_hieght),pad_with=pad_with):
        # sort the element in the output window
        sorted_flat_neighbors = [elem for rows in neighbors for elem in rows]
        sorted_flat_neighbors.sort()
        # change the pixel in the original image
        output_mat[center[0]][center[1]] = sorted_flat_neighbors[rank - 1]
    return output_mat

def outlier_method(matrix: Matrix, threshold:float) -> Matrix:
    """
    Applies an outlier detection and correction method to a matrix based on a threshold.
    If the mean value of a pixel's neighborhood exceeds the threshold, the pixel value is
    replaced with the rounded neighborhood mean. This helps to smooth out isolated outliers
    in the matrix.

    Parameters:
        `matrix` (Matrix): The original matrix on which the outlier method is applied.
        `threshold` (float): The threshold value used for determining outliers.

    Returns:
        Matrix: A new matrix with outliers replaced by neighborhood means where applicable.
    
    Notes
    -----
    - The 3x3 mean kernel used in this method is defined as:
        mean_kernel = [[0.125, 0.125, 0.125], 
                       [0.125, 0, 0.125], 
                       [0.125, 0.125, 0.125]]
    - The kernel excludes the center pixel from the mean computation.
    - Only pixels with a neighborhood mean greater than or equal to the threshold
      are updated to the neighborhood mean.

    Examples:
    >>> img = [[10, 20, 30],
    ...        [40, 50, 60],
    ...        [70, 80, 90]]
    >>> processed_img = outlier_method(img, threshold=50)
    """

    # Deep copy the input matrix to avoid modifying the original matrix
    output_mat = deepcopy(matrix)

    # Define the 3x3 mean kernel for neighborhood calculation
    # for the neighberhood multiply by 1/8 in all places but the center
    mean_kernel = [[0.125, 0.125, 0.125], [0.125, 0, 0.125], [0.125, 0.125, 0.125]]
    k_height = len(mean_kernel)
    k_width = len(mean_kernel[0])

    # Iterate through each neighborhood with its center pixel
    for neighbors, center in _get_neighborhood_and_center(output_mat, (k_width, k_height)):
        # Calculate the mean value of the neighborhood using the mean kernel
        neighborhood_mean = sum(neighbors[row][col] * mean_kernel[row][col] for row in range(3) for col in range(3))

        # Replace the center pixel with the rounded neighborhood mean if it exceeds the threshold
        if neighborhood_mean >= threshold:
            output_mat[center[0]][center[1]] = DP.normal_round(neighborhood_mean)

    return output_mat

def adaptive_filter(matrix, kernel_dim:Tuple[int,int], pad_with:Optional[int] = None) -> Matrix:
    """
    Applies adaptive local noise reduction filter using local statistics. The filter adapts to local image properties

    Parameters:
        `matrix` (Matrix): Input image as 2D list
        `kernel_dim` (Tuple[int, int]): (width, height) of the neighborhood window
        `pad_with` (int, optional): Padding value for border handling

    Returns:
        Matrix: Filtered image with noise reduction

    steps:
    - calculate global mean `mu_g` to calculate population variance `sigma_g`
    - calculate local mean `mu` and sample variance `sigma`
    - get the estimate value of of pixel (x,y)
    - pixel_est = mu + sigma * (pixel - mu)/(sigma + sigma_g)
    """
    def calculate_statistics(values) -> Tuple[float, float]:
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n - 1
        return mean, variance

    k_width = kernel_dim[0]
    k_hieght = kernel_dim[1]
    kernel_size = k_width * k_hieght

    output_mat = deepcopy(matrix)



    # calculate population variance `sigma_g`
    flat_mat = flatten_matrix(matrix)
    mat_size = len(flat_mat)
    mu_g = sum(flat_mat) / mat_size
    sigma_g = sum((x - mu_g) ** 2 for x in flat_mat) / mat_size

    for neighbors, center in _get_neighborhood_and_center(matrix,(k_width,k_hieght),pad_with=pad_with):
        # calculate local mean `mu` and sample variance `sigma`
        mu = sum(elem for rows in neighbors for elem in rows)/kernel_size
        sigma = sum((elem - mu)**2 for rows in matrix for elem in rows)/(kernel_size-1)

        current_pixel = output_mat[center[0]][center[1]]

        output_mat[center[0]][center[1]] = DP.normal_round(mu + sigma * (current_pixel - mu)/(sigma + sigma_g))

    return output_mat

def apply_filter(matrix: Matrix, kernel: Matrix, pad_with:Optional[int] = None, bit_depth:Optional[int] = None) -> Matrix:
    """wrapper on correlate function that round the output"""
    
    return [[DP.normal_round(v) for v in row] for row in correlate(matrix, kernel, pad_with, bit_depth)]

def create_gaussian_kernel(size: int, sigma: float) -> List[List[float]]:
    """
    Create a 2D Gaussian kernel (filter) for image smoothing.
    
    The Gaussian kernel is created using the 2D Gaussian function:
    G(x,y) = (1/(2π*σ²)) * e^(-(x² + y²)/(2σ²))
    
    Where:
    - x,y are distances from the kernel center
    - σ (sigma) controls the spread of the Gaussian
    - Size determines the kernel dimensions (must be odd)
    
    Parameters:
        size: Kernel size (must be odd). Larger size = more pixels involved in smoothing
        sigma: Standard deviation of Gaussian. Larger sigma = more blurring
    
    Returns:
        2D list representing the Gaussian kernel, normalized to sum to 1
        
    Raises:
        ValueError: If size is even or sigma is not positive
    """
    from math import exp, pi

    # Validate inputs
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd to have a clear center point")
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    
    # Calculate center point of kernel
    center = size // 2
    
    # Calculate normalization factor: 1/(2π*σ²)
    # This is part of the 2D Gaussian formula
    norm_factor = 1 / (2 * pi * sigma * sigma)
    
    # Initialize kernel and running sum for normalization
    kernel = []
    kernel_sum = 0.0
    
    # Generate kernel values
    for i in range(size):
        row = []
        for j in range(size):
            # Calculate distance from center (x,y coordinates relative to center)
            x = i - center
            y = j - center
            
            # Calculate squared distance from center
            # This is (x² + y²) in the formula
            squared_dist = x*x + y*y
            
            # Calculate exponent: -(x² + y²)/(2σ²)
            exponent = -squared_dist / (2 * sigma * sigma)
            
            # Calculate Gaussian value using the complete formula
            value = norm_factor * exp(exponent)
            
            # Add to running sum for later normalization
            kernel_sum += value
            row.append(value)
        kernel.append(row)
    
    # Normalize kernel so all values sum to 1
    # This ensures image brightness is preserved after filtering
    for i in range(size):
        for j in range(size):
            kernel[i][j] = round(kernel[i][j] / kernel_sum,4)
            
    return kernel

# -------------------- Edge Detection --------------------------

def apply_gradient_operator(
    matrix: Matrix,
    kernel: Matrix,
    bit_depth: Optional[int] = None,
    pad_with: Optional[int] = None
) -> Matrix:
    """
    Generic function to apply gradient operators using pure Python.
    
    Parameters::
        `matrix`: Input image as 2D list
        `kernel`: Primary gradient kernel
        `bit_depth`: Bit depth for output value clipping
        `pad_with`: Padding value for border handling
        
    Returns:
        Matrix: Edge-detected image
    """
    rotated_kernel = rotate_90(kernel)
    
    gradient_x = correlate(matrix, kernel, pad_with, bit_depth) #f1
    gradient_y = correlate(matrix, rotated_kernel, pad_with, bit_depth) #f2
    
    height, width = len(gradient_x), len(gradient_x[0])
    result = [[0] * width for _ in range(height)]
    
    for r in range(height):
        for c in range(width):
            magnitude = m.sqrt(gradient_x[r][c]**2 + gradient_y[r][c]**2)
            result[r][c] = DP.normal_round(magnitude)
    
    return result

def roberts_operator(
    matrix: Matrix,
    bit_depth: Optional[int] = None,
    pad_with: Optional[int] = None
) -> Matrix:
    """
    Applies Roberts cross-gradient operator for edge detection.

    Parameters:
        `matrix` (Matrix): Input image as 2D list
        `bit_depth` (int, optional): Bit depth for output value clipping
        `pad_with` (int, optional): Padding value for border handling

    Returns:
        Matrix: Edge-detected image

    Notes
    -----
    The Roberts operator uses two 2×2 kernels to compute diagonal gradients:
    Gx = [[-1, 0],     Gy = [[0, -1],
          [0, 1]]           [1, 0]]
    
    The final magnitude is computed as: sqrt(Gx² + Gy²)

    Examples:
    >>> img = [[1, 2, 3],
    ...        [4, 5, 6],
    ...        [7, 8, 9]]
    >>> edges = roberts_operator(img, pad_with=0)
    >>> # Result highlights diagonal edges
    """
    return apply_gradient_operator(matrix, Kernels.robert_operator, bit_depth, pad_with)

def prewitt_operator(matrix: Matrix, bit_depth: Optional[int] = None, pad_with: Optional[int] = None) -> Matrix:
    """
    Applies the Prewitt operator for edge detection.

    Parameters:
        `matrix` (Matrix): Input image as a 2D list
        `bit_depth` (int, optional): Bit depth for output value clipping
        `pad_with` (int, optional): Padding value for border handling

    Returns:
        Matrix: Edge-detected image

    Notes
    -----
    The Prewitt operator uses two 3×3 kernels to compute horizontal and vertical gradients:
    Gx = [[-1, 0, 1],     Gy = [[-1, -1, -1],
          [-1, 0, 1],           [ 0,  0,  0],
          [-1, 0, 1]]           [ 1,  1,  1]]
    
    The final magnitude is computed as: sqrt(Gx² + Gy²)

    Examples:
    >>> img = [[1, 2, 3],
    ...        [4, 5, 6],
    ...        [7, 8, 9]]
    >>> edges = prewitt_operator(img, pad_with=0)
    >>> # Result highlights vertical and horizontal edges
    """
    return apply_gradient_operator(matrix, Kernels.prewitt_operator, bit_depth, pad_with)

def sobel_operator(
    matrix: Matrix,
    bit_depth: Optional[int] = None,
    pad_with: Optional[int] = None
) -> Matrix:
    """
    Applies the Sobel operator for edge detection with enhanced sensitivity.

    Parameters:
        matrix (Matrix): Input image as a 2D list.
        bit_depth (int, optional): Bit depth for output value clipping.
        pad_with (int, optional): Padding value for border handling.

    Returns:
        Matrix: Edge-detected image.

    Notes:
        The Sobel operator uses two 3×3 kernels to compute horizontal and vertical gradients:
        
        Gx = [[-1, 0, 1],     Gy = [[-1, -2, -1],
            [-2, 0, 2],           [ 0,  0,  0],
            [-1, 0, 1]]           [ 1,  2,  1]]
        
        The final magnitude is computed as: sqrt(Gx² + Gy²).

    Examples:
        >>> img = [[1, 2, 3],
        ...        [4, 5, 6],
        ...        [7, 8, 9]]
        >>> edges = sobel_operator(img, pad_with=0)
        >>> # Result highlights vertical and horizontal edges.
    """
    return apply_gradient_operator(matrix, Kernels.sobel_operator, bit_depth, pad_with)
