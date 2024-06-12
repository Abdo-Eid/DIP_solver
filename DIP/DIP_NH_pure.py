from DIP.helper import *
import DIP.DIP_pure as DP

class Kernals:
    smoothing_filter = [[0.111,0.111,0.111],
                        [0.111,0.111,0.111],
                        [0.111,0.111,0.111]] # 1/9
    
    weighted_smoothing_filter = [[0.0625,0.125,0.0625],
                                 [0.125,0.25,0.125],
                                 [0.0625,0.125,0.0625]]
    
    high_pass_filter = [[-1,-1,-1],
                        [-1,8,-1],
                        [-1,-1,-1]]
    
    # for first drivative operators it's a filter then it's 90deg rotated then the result is their eculodian distance 
    robert_operator = [[-1,0],
                       [0,1]]
    
    prewitt_operator = [[-1,0,1],
                        [-1,0,1],
                        [-1,0,1]]
    
    sobel_operator = [[-1,0,1],
                      [-2,0,2],
                      [-1,0,1]]
    
    # second drivative is like normal filter
    laplacian_operator = [[0,1,0],
                          [1,-4,1],
                          [0,1,0]]

# -------------------------------------------

def pad(matrix: Matrix, kernel_dim: Tuple[int,int],pad_with: int = None):
    """
    Pad a matrix with a specified value or return a deep copy of the matrix.

    Parameters:
    - `matrix` (Matrix): The original matrix to be padded.
    - `kernel_dim` (Tuple[int,int]): (`width`,`height`) The kernel dimensions used to determine the padding.
    - `pad_with` (int, optional): The value to pad the matrix with. If `None`, a deep copy of the original matrix is returned. Defaults to `None`.

    Returns:
    - `Matrix`: The padded matrix or a deep copy of the original matrix if `pad_with` is `None`.

    Description:
    This function pads the given `matrix` with a specified integer value based on the dimensions of the `kernel`. If `pad_with` is `None`, it returns a deep copy of the original matrix. The padding is calculated to center the kernel on the matrix, adding rows and columns as needed. The function raises a `ValueError` if `pad_with` is not `None` or an integer.
    """

    if pad_with == None:
        return deepcopy(matrix)
    elif isinstance(pad_with,int):
        width = len(matrix[0])
        height = len(matrix)

        k_w = kernel_dim[0]
        k_h = kernel_dim[1]
        # get the how much center is away from each corner of the kernal
        # the index is number of rows or column away from this point
        pad_left,pad_top = DP.get_center(k_w,k_h)
        # lenght - indx = lenght - 1 = indx
        pad_right = k_w - pad_left - 1
        pad_down = k_h - pad_top - 1

        # make bigger matrix 
        pad_h = height + pad_top + pad_down
        pad_w = width + pad_left + pad_right
        pad_matrix = [[pad_with for _ in range(pad_w)] for _ in range(pad_h)]

        # loop on the old matrix to copy the values with offset of kernel center
        for r in range(height):
            for c in range(width):
                pad_matrix[r + pad_top][c + pad_left] = matrix[r][c]
        
        return pad_matrix
    else:
        raise ValueError("padding must be None or integer")

def _neighborhoods(matrix: Matrix, kernel_dim: Tuple[int,int]) -> Generator[Matrix, None, None]:
    """
    Generate neighborhoods of a matrix based on a given kernel.

    Parameters:
    - `matrix` (Matrix): The original matrix from which neighborhoods are extracted.
    - `kernel_dim` (Tuple[int,int]): (`width`,`height`) The kernel dimensions used to determine the size and shape of neighborhoods.

    Returns:
    - `Generator[Matrix, None, None]`: A generator that yields matrices representing the neighborhoods of the original matrix based on the kernel size.

    Description:
    This function generates submatrices (neighborhoods) from the original `matrix` with dimensions based on the `kernel`. The neighborhoods are extracted by sliding the kernel across the matrix. For each position, a submatrix of the same size as the kernel is yielded. The kernel is centered on each position in the matrix, ensuring that only valid submatrices within the matrix boundaries are generated.

    Example:
    ```python
    >>> matrix = [[1, 2, 3],
    ...           [5, 6, 7],
    ...           [9, 10, 11]]
    >>> kernel_dim = (2, 2)
    >>> gen = _neighborhoods(matrix, kernel_dim)
    >>> for neighborhood in gen:
    ...     print(neighborhood)
    [[1, 2], [5, 6]]
    [[2, 3], [6, 7]]
    [[5, 6], [9, 10]]
    [[6, 7], [10, 11]]
    """

    width = len(matrix[0])
    height = len(matrix)

    k_w = kernel_dim[0]
    k_h = kernel_dim[1]
    k_center = DP.get_center(k_w,k_h)

    # going from 0 to the last place the kernel center can be located in so the filter is in the image
    for row in range(0,height - k_h + k_center[0]):
        for col in range(0,width - k_w + k_center[1]):
            # getting the values same size as the filter 
            neighborhood_matrix = [r[col:col + k_w] for r in matrix[row:row + k_h]]
            yield neighborhood_matrix

def _neighborhoods_of_pixel(matrix: Matrix, pixel_pos: Tuple[int, int], kernel_dim: Tuple[int, int], pad_with: int = 0) -> Matrix:
    """
    Extract a neighborhood of a pixel from a matrix based on a given kernel.

    Parameters:
    - `matrix` (Matrix): The original matrix from which the neighborhood is extracted.
    - `pixel_pos` (Tuple[int, int]): The position of the pixel for which the neighborhood is extracted.
    - `kernel_dim` (Tuple[int, int]): (`width`, `height`) The dimensions of the kernel used to determine the size and shape of the neighborhood.
    - `pad_with` (int, optional): The value to pad the neighborhood with if it extends beyond the matrix boundaries. Defaults to 0.

    Returns:
    - `Matrix`: A matrix representing the neighborhood of the specified pixel based on the kernel size.

    Description:
    This function extracts a neighborhood of a pixel from the original `matrix` based on the dimensions of the `kernel`. It pads the neighborhood with the specified value if it extends beyond the boundaries of the matrix. The neighborhood is centered on the specified `pixel_pos`. The function calculates the starting and ending positions of the neighborhood within the matrix, creates a zero-filled matrix of the kernel shape, calculates the shift amount to align the neighborhood with the kernel, and copies values from the original matrix to the output matrix.

    Example:
    ```python
    >>> matrix = [[1, 2, 3, 4],
    ...           [5, 6, 7, 8],
    ...           [9, 10, 11, 12],
    ...           [13, 14, 15, 16]]
    >>> pixel_pos = (1, 1)
    >>> kernel_dim = (3, 3)
    >>> neighborhood = _neighborhoods_of_pixel(matrix, pixel_pos, kernel_dim)
    >>> neighborhood
    [[1, 2, 3],
    [5, 6, 7],
    [9, 10, 11]]
    """

    width = len(matrix[0])
    height = len(matrix)

    k_w = kernel_dim[0]
    k_h = kernel_dim[1]
    left,top = DP.get_center(k_w,k_h)
    # lenght - indx = lenght - 1 = indx
    right = k_w - left - 1
    down = k_h - top - 1

    output_matrix = [[pad_with for _ in range(k_w)] for _ in range(k_h)]

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

def _neighborhood_with_center(matrix: Matrix, kernel_dim: Tuple[int,int]) -> Generator[Matrix,None,None]:
    """
    Generate neighborhoods of a matrix based on a given kernel, including the center pixel coordinates.

    Parameters:
    - `matrix` (Matrix): The original matrix from which neighborhoods are extracted.
    - `kernel_dim` (Tuple[int,int]): (`width`, `height`) The kernel dimensions used to determine the size and shape of neighborhoods.

    Yields:
    - `Tuple[Matrix, Tuple[int,int]]`: A tuple containing the neighborhood matrix and the coordinates of its center pixel.

    Description:
    This generator function generates submatrices (neighborhoods) from the original `matrix` with dimensions based on the `kernel`. For each position where the kernel can fit within the matrix, a submatrix of the same size as the kernel is yielded along with the coordinates of its center pixel. The kernel is centered on each position in the matrix, ensuring that only valid submatrices within the matrix boundaries are generated.

    The generator yields tuples where the first element is the neighborhood matrix and the second element is a tuple containing the coordinates of the center pixel of the neighborhood.

    Example:
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

    width = len(matrix[0])
    height = len(matrix)

    k_w = kernel_dim[0]
    k_h = kernel_dim[1]
    k_center = DP.get_center(k_w,k_h)

    # going from 0 to the last place the kernel center can be located in so the filter is in the image
    for row in range(0,height - k_h + k_center[0]):
        for col in range(0,width - k_w + k_center[1]):
            # getting the values same size as the filter 
            neighbors_with_center = [r[col:col + k_w] for r in matrix[row:row + k_h]],(row+k_center[0],col + k_center[1])
            yield neighbors_with_center

def convolute(matrix: Matrix, kernel: Matrix,pad_with:int = None,bit_depth:int = 8) -> Matrix:
    """
    Perform convolution operation on a matrix using a given kernel.

    Parameters:
    - `matrix` (Matrix): The original matrix on which convolution is applied.
    - `kernel` (Matrix): The convolution kernel.
    - `pad_with` (int, optional): The value used for padding the matrix. If `None`, no padding is applied. Defaults to `None`.

    Returns:
    - `Matrix`: The result of convolving the matrix with the kernel.

    Description:
    This function performs convolution operation on the `matrix` using the specified `kernel`. It supports optional padding with a specified value. If `pad_with` is `None`, no padding is applied, and the output matrix is smaller than the input matrix. If `pad_with` is an integer, the matrix is padded with the specified value. Convolution is performed by sliding the kernel over the matrix and computing the dot product between the kernel and the neighborhood of each pixel. The result is stored in the output matrix.

    Example:
    ```python
    >>> matrix = [[1, 2, 3],
    ...           [4, 5, 6],
    ...           [7, 8, 9]]
    >>> kernel = [[1, 0, -1],
    ...           [1, 0, -1],
    ...           [1, 0, -1]]
    >>> result = convolute(matrix, kernel,pad_with=0)
    >>> result
    [[-7, -4, 7],
    [-15, -6, 15],
    [-13, -4, 13]]
    """

    bit_depth = bit_depth_from_mat(matrix)
    max_value = 2 ** bit_depth - 1
    
    height = len(matrix)
    width = len(matrix[0])
    k_w = len(kernel[0])
    k_h = len(kernel)
    # get the how much center is away from each corner of the kernal
    # the index is number of rows or column away from this point
    left,top = DP.get_center(k_w,k_h)
    # lenght - indx = lenght - 1 = indx
    right = k_w - left - 1
    down = k_h - top - 1

    if pad_with == None:
        # make smaller matrix
        height = height - top - down
        width = width - left - right
        output_mat = [[pad_with for _ in range(width)] for _ in range(height)]
    elif isinstance(pad_with,int):
        # output will be same size as original in this case
        output_mat = [[pad_with for _ in range(width)] for _ in range(height)]
    

        # make bigger padded matrix 
        pad_h = height + top + down
        pad_w = width + left + right
        pad_matrix = [[pad_with for _ in range(pad_w)] for _ in range(pad_h)]

        # loop on the old matrix to copy the values with offset of kernel center
        for r in range(height):
            for c in range(width):
                pad_matrix[r + top][c + left] = matrix[r][c]
        
        matrix = pad_matrix
    else:
        raise ValueError("padding must be None or integer")

    # going from 0 to the last place the kernel center can be located in so the filter is in the image
    for row in range(0,height - down + 1 ):
        for col in range(0,width - right + 1):
            # getting the values same size as the filter 
            neighborhood = [r[col:col + k_w] for r in matrix[row:row + k_h]]
            # for every row in neighborhood and kernal and for each element in both rows multiply elements then sum all
            output_mat[row][col] = clip(round(sum(e * v for n_row, ker_row in zip(neighborhood, kernel) for e,v in zip(n_row,ker_row)),2),0,max_value)

    return output_mat

def rank_order_filter(matrix, kernel_dim:Tuple[int,int], rank:int, pad_with:int = None) -> Matrix:
    """
    Apply rank-order filter on a matrix using a given kernel size and rank.

    Parameters:
    - `matrix` (Matrix): The original matrix on which the filter is applied.
    - `kernel_dim` (Tuple[int,int]): (`width`, `height`) The dimensions of the filter kernel.
    - `rank` (int): The rank used in the rank-order filter, starts from 1 to the size of the kernal (width * height).
    - `pad_with` (int, optional): The value used for padding the matrix. If `None`, no padding is applied. Defaults to `None`.

    Returns:
    - `Matrix`: The result of applying the rank-order filter on the matrix.

    Description:
    This function applies a rank-order filter on the `matrix` using a filter kernel of specified dimensions (`kernel_dim`). The `rank` parameter determines which element in the sorted neighborhood values is selected as the output pixel value. If `pad_with` is provided, the matrix is padded with the specified value. The output matrix has the same size as the input matrix.

    Example:
    ```python
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

    # so if the matrix is padded we start from the original matrix start
    offset_w,offset_h = 0,0
    output_mat = deepcopy(matrix)
    
    if pad_with == None:
        matrix = deepcopy(matrix)
    elif isinstance(pad_with,int):
        matrix = pad(matrix,(k_width,k_hieght),pad_with)
        # subtract the added offset to get the original matrix
        offset_w,offset_h = DP.get_center(k_width,k_hieght)
    else:
        raise ValueError("padding must be None or integer")

    for neighbers, center in _neighborhood_with_center(matrix,(k_width,k_hieght)):
        # sort the element in the output window
        sorted_flat_neighbers = [elem for rows in neighbers for elem in rows]
        sorted_flat_neighbers.sort()
        # change the pixel in the original image
        output_mat[center[0] - offset_h][center[1]- offset_w] = sorted_flat_neighbers[rank - 1]
    return output_mat

def outlier_method(matrix: Matrix, threshold:float) -> Matrix:
    """
    Apply the outlier method on a matrix using a threshold value.

    Parameters:
    - `matrix` (Matrix): The original matrix on which the outlier method is applied.
    - `threshold` (float): The threshold value used for determining outliers.

    Returns:
    - `Matrix`: The result of applying the outlier method on the matrix.

    Description:
    This function applies the outlier method on the `matrix` using a specified threshold value. The outlier method replaces each pixel with the mean value of its neighborhood if the mean value is greater than or equal to the threshold. The neighborhood mean is calculated using a 3x3 mean kernel with weights distributed equally to all neighbors except the center pixel. The output matrix has the same size as the input matrix.
    """

    # Deep copy the input matrix to avoid modifying the original matrix
    output_mat = deepcopy(matrix)

    # Define the 3x3 mean kernel for neighborhood calculation
    # for the neighberhood multiply by 1/8 in all places but the center
    mean_kernel = [[0.125, 0.125, 0.125], [0.125, 0, 0.125], [0.125, 0.125, 0.125]]
    k_height = len(mean_kernel)
    k_width = len(mean_kernel[0])

    # Iterate through each neighborhood with its center pixel
    for neighbors, center in _neighborhood_with_center(output_mat, (k_width, k_height)):
        # Calculate the mean value of the neighborhood using the mean kernel
        neighborhood_mean = sum(neighbors[row][col] * mean_kernel[row][col] for row in range(3) for col in range(3))
        
        # Replace the center pixel with the rounded neighborhood mean if it exceeds the threshold
        if neighborhood_mean >= threshold:
            output_mat[center[0]][center[1]] = DP.custom_round(neighborhood_mean)
    
    return output_mat

def adaptive_filter(matrix, kernel_dim:Tuple[int,int], pad_with:int = None) -> Matrix:
    """
    steps
    - calculate local mean `mu` and sample variance `sigma`
    - calculate global mean `mu_g` and population variance `sigma_g`
    - get the estimate value of of pixel (x,y)
    - pixel_est = mu + sigma * (pixel - mu)/(sigma + sigma_g)
    """
    
    k_width = kernel_dim[0]
    k_hieght = kernel_dim[1]
    kernel_size = k_width * k_hieght
    matrix_size = len(matrix) * len(matrix[0])

    # so if the matrix is padded we start from the original matrix start
    offset_w,offset_h = 0,0
    output_mat = deepcopy(matrix)
    
    if pad_with == None:
        matrix = deepcopy(matrix)
    elif isinstance(pad_with,int):
        matrix = pad(matrix,(k_width,k_hieght),pad_with)
        # subtract the added offset to get the original matrix
        offset_w,offset_h = DP.get_center(k_width,k_hieght)
    else:
        raise ValueError("padding must be None or integer")
    
    # calculate global mean `mu_g` and population variance `sigma_g`
    mu_g = sum(elem for rows in matrix for elem in rows)/matrix_size
    sigma_g = sum((elem - mu_g)**2 for rows in matrix for elem in rows)/matrix_size

    for neighbers, center in _neighborhood_with_center(matrix,(k_width,k_hieght)):
        # calculate local mean `mu` and sample variance `sigma`
        mu = sum(elem for rows in neighbers for elem in rows)/kernel_size
        sigma = sum((elem - mu)**2 for rows in matrix for elem in rows)/(kernel_size-1)

        current_pixel = output_mat[center[0] - offset_h][center[1]- offset_w]

        output_mat[center[0] - offset_h][center[1]- offset_w] = DP.custom_round(mu + sigma * (current_pixel - mu)/(sigma + sigma_g))

    return output_mat

# -------------------- Edge Detection --------------------------

def _rotate_90_clockwise(matrix: Matrix) -> Matrix:
    if not matrix:
        return []

    rows = len(matrix)
    cols = len(matrix[0])
    
    # Transpose the matrix
    transposed = [[matrix[j][i] for j in range(rows)] for i in range(cols)]
    
    # Reverse each row to get the clockwise rotation
    clockwise_rotated = [list(reversed(row)) for row in transposed]
    
    return clockwise_rotated

def robert_operator():
    pass

def prewitt_operator():
    pass

def sobel_operator():
    pass

# -------------------- morphology --------------------------

def dilation(matrix: Matrix, SE: Matrix) -> Matrix:
    """
    Perform dilation operation on a matrix using a given structuring element (SE).

    Parameters:
    - `matrix` (Matrix): The original binary matrix on which the dilation is applied.
    - `SE` (Matrix): The structuring element used for the dilation operation.

    Returns:
    - `Matrix`: The result of the dilation operation on the matrix.

    Description:
    This function performs a dilation operation on the `matrix` using the specified structuring element `SE`. The dilation process involves sliding the SE over the matrix and checking if any '1' pixel in the SE aligns with a '1' pixel in the matrix neighborhood. If such an overlap is found, the center pixel of the neighborhood is set to '1' in the output matrix; otherwise, it remains '0'. The output matrix has the same size as the input matrix.

    Example:
    ```python
    >>> matrix = [[1, 0, 0, 0],
    ...           [0, 0, 0, 0],
    ...           [0, 0, 0, 0],
    ...           [0, 0, 0, 1]]
    >>> SE = [[1, 1],
    ...       [1, 1]]
    >>> result = dilation(matrix, SE)
    >>> result
    [[1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1]]
    """
    height = len(matrix)
    width = len(matrix[0])
    SE_w = len(SE[0])
    SE_h = len(SE)

    # subtract the added offset to get the original matrix
    # we're sliding on the padded image, but we need the center in the original image
    offset_w,offset_h = DP.get_center(SE_w,SE_h)

    # output will be same size as original in this case
    output_mat = [[0 for _ in range(width)] for _ in range(height)]

    # make bigger padded matrix 
    matrix = pad(matrix,(SE_w,SE_h),pad_with=0)

    for neighbers, center in _neighborhood_with_center(matrix,(SE_w,SE_h)):

        overlap_found = False

        # loop on each element of the strucure and the neighbers
        for row in range(SE_h):
            for col in range(SE_w):
                if SE[row][col] and neighbers[row][col]:
                    overlap_found = True
                    break
            if overlap_found:
                break
        if overlap_found:
            output_mat[center[0] - offset_h][center[1]- offset_w] = 1

    return output_mat

def erosion(matrix: Matrix, SE: Matrix) -> Matrix:
    """
    Perform erosion operation on a matrix using a given structuring element (SE).

    Parameters:
    - `matrix` (Matrix): The original binary matrix on which the erosion is applied.
    - `SE` (Matrix): The structuring element used for the erosion operation.

    Returns:
    - `Matrix`: The result of the erosion operation on the matrix.

    Description:
    This function performs an erosion operation on the `matrix` using the specified structuring element `SE`. The erosion process involves sliding the SE over the matrix and checking if all '1' pixels in the SE align with '1' pixels in the matrix neighborhood. If they do, the center pixel of the neighborhood is set to '1' in the output matrix; otherwise, it is set to '0'. The output matrix has the same size as the input matrix.

    Example:
    ```python
    >>> matrix = [[1, 1, 0, 0],
    ...           [1, 1, 0, 0],
    ...           [0, 0, 1, 1],
    ...           [0, 0, 1, 1]]
    >>> SE = [[1, 1],
    ...       [1, 1]]
    >>> result = erosion(matrix, SE)
    >>> result
    [[0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1]]
    """
    height = len(matrix)
    width = len(matrix[0])
    SE_w = len(SE[0])
    SE_h = len(SE)

    # subtract the added offset to get the original matrix
    # we're sliding on the padded image, but we need the center in the original image
    offset_w,offset_h = DP.get_center(SE_w,SE_h)

    # output will be same size as original in this case
    output_mat = [[0 for _ in range(width)] for _ in range(height)]

    # make bigger padded matrix 
    matrix = pad(matrix,(SE_w,SE_h),pad_with=0)

    for neighbers, center in _neighborhood_with_center(matrix,(SE_w,SE_h)):

        fits = True

        # loop on each element of the strucure and the neighbers
        for row in range(SE_h):
            for col in range(SE_w):
                # For each position of the SE, check if all '1' pixels in the SE align with '1' pixels in the image.
                # so if 1 and 0 will enter the if
                if SE[row][col] and not neighbers[row][col]:
                    fits = False
                    break
            if not fits:
                break
        if fits:
            output_mat[center[0] - offset_h][center[1]- offset_w] = 1

    return output_mat

def opening(matrix: Matrix, SE: Matrix) -> Matrix:
    """
    Perform opening operation on a matrix using a given structuring element (SE).

    Parameters:
    - `matrix` (Matrix): The original binary matrix on which the opening operation is applied.
    - `SE` (Matrix): The structuring element used for the opening operation.

    Returns:
    - `Matrix`: The result of the opening operation on the matrix.

    Description:
    This function performs an opening operation on the `matrix` using the specified structuring element `SE`. The opening process consists of an erosion followed by a dilation using the same structuring element. This operation is useful for removing small objects from the foreground (usually taken as the bright pixels) of an image, placing them in the background, while preserving the shape and size of larger objects in the image.

    Example:
    >>> matrix = [[0,0,0,0,0,0], 
    ...          [0,1,1,1,1,0], 
    ...          [0,1,1,0,1,0],
    ...          [0,1,1,0,1,0],
    ...          [0,1,1,1,1,0]]
    >>> SE = [[1,1]
             ,[0,1]]
    >>> result = opening(matrix,SE)
    >>> result
        [[0, 0, 0, 0, 0, 0]
        ,[0, 0, 0, 0, 0, 0]
        ,[0, 0, 1, 0, 1, 0]
        ,[0, 0, 1, 1, 1, 1]
        ,[0, 0, 1, 1, 0, 0]]
    """
    eroded = erosion(matrix, SE)
    result = dilation(eroded, SE)
    return result

def closing(matrix: Matrix, SE: Matrix) -> Matrix:
    """
    Perform closing operation on a matrix using a given structuring element (SE).

    Parameters:
    - `matrix` (Matrix): The original binary matrix on which the closing operation is applied.
    - `SE` (Matrix): The structuring element used for the closing operation.

    Returns:
    - `Matrix`: The result of the closing operation on the matrix.

    Description:
    This function performs a closing operation on the `matrix` using the specified structuring element `SE`. The closing process consists of a dilation followed by an erosion using the same structuring element. This operation is useful for closing small holes in the foreground, bridging small gaps, and generally smoothing the outline of objects while keeping their sizes and shapes approximately the same.

    Example:
    >>> matrix = [[0,0,0,0,0,0], 
    ...          [0,1,1,1,1,0], 
    ...          [0,1,1,0,1,0],
    ...          [0,1,1,0,1,0],
    ...          [0,1,1,1,1,0]]
    >>> SE = [[1,1]
             ,[0,1]]
    >>> result = closing(matrix,SE)
    >>> result
        [[0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0], 
        [0, 0, 1, 1, 1, 0], 
        [0, 0, 1, 1, 1, 1], 
        [0, 0, 1, 1, 1, 1]]
    """
    dilated = dilation(matrix, SE)
    result = erosion(dilated, SE)
    return result

def internal_extraction(matrix: Matrix, SE: Matrix) -> Matrix:

    eroded_matrix = erosion(matrix, SE)

    # Subtract the eroded image from the original image to get the external boundary
    result_matrix = [[DP.clip(matrix[i][j] - eroded_matrix[i][j],0,1) for j in range(len(matrix[0]))] for i in range(len(matrix))]
    
    return result_matrix

def external_extraction(matrix: Matrix, SE: Matrix) -> Matrix:

    dilated_matrix = dilation(matrix, SE)
    
    # Subtract the original image from the dilated image to get the external boundary
    result_matrix = [[DP.clip(dilated_matrix[i][j] - matrix[i][j],0,1) for j in range(len(matrix[0]))] for i in range(len(matrix))]
    
    return result_matrix

def morphology_gradient(matrix: Matrix, SE: Matrix) -> Matrix:

    dilated_matrix = dilation(matrix, SE)

    eroded_matrix = erosion(matrix, SE)
    
    result_matrix = [[abs(dilated_matrix[i][j] - eroded_matrix[i][j]) for j in range(len(matrix[0]))] for i in range(len(matrix))]
    
    return result_matrix

def hole_filling(matrix: Matrix, SE: Matrix, x_0_indx: Tuple[int,int],max_itiration:int = 100) -> Matrix:

    X_prev = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    X_prev[x_0_indx[0]][x_0_indx[1]] = 1

    complement_A = DP.complement(matrix,1)

    for _ in range(max_itiration):
        # Compute X_k = (X_{k-1} ⊕ B) ∩ A^c
        dilated_X_prev  = dilation(X_prev, SE)
        X_k = intersection(dilated_X_prev, complement_A)

        # Check for convergence
        if X_k == X_prev:
            break

        # Update X_prev for next iteration
        X_prev = X_k

    # Combine filled hole with original foreground
    result = union(X_k, matrix)

    return result

def hit_or_miss(matrix: Matrix, SE1: Matrix, SE2: Matrix) -> Matrix:

    complement_A = DP.complement(matrix,1)

    A_erosion = erosion(matrix,SE1)
    A_c_dilation = erosion(complement_A,SE2)
    result = intersection(A_erosion,A_c_dilation)
    return result
