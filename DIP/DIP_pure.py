
from .helper import *


def resize(mat: Matrix, size: Tuple[int, int]) -> Matrix:
    """
    Resize a matrix using a custom algorithm.

    Parameters:
    - `mat` (Matrix): The source matrix to be resized.
    - `size` (Tuple[int, int]): A tuple specifying the dimensions (`dst_rows`, `dst_cols`) of the resized matrix.

    Returns:
    - `Matrix`: The resized matrix.

    Description:
    This function resizes a given matrix `mat` to the specified dimensions `size` using a custom algorithm. It spreads the original values from the source matrix into the resized matrix, placing empty cells in between the original values. It then fills in the empty cells based on the values of their neighboring cells.

    Steps:
    1. Calculate the number of empty cells between the original values.
    2. Create a new resized matrix with the specified dimensions, initially filled with zeros.
    3. Spread the original values into the new matrix at appropriate intervals.
    4. Fill in the empty cells:
    - For empty row cells, take the average of the cells above and below.
    - For empty column cells, take the average of the cells to the left and right.
    - For empty diagonal cells, take the average of the four neighboring diagonal cells.

    Example:
    >>> mat = [
    ...     [1, 2, 3],
    ...     [4, 5, 6],
    ...     [7, 8, 9]
    ... ]
    >>> size = (6, 6)
    >>> resized_matrix = resize(mat, size)
    >>> for row in resized_matrix:
    ...     print(row)
    [1, 1, 2, 2, 3, 3]
    [2, 2, 3, 3, 4, 4]
    [4, 4, 5, 5, 6, 6]
    [5, 5, 6, 6, 7, 7]
    [7, 7, 8, 8, 9, 9]
    [8, 8, 9, 9, 10, 10]
    """
    src_rows, src_cols = len(mat), len(mat[0])
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
            resized_mat[(row_spacing + 1) * i][(col_spacing + 1) * j] = mat[i][j]

    # Fill in the empty cells
    for i in range(dst_rows):
        for j in range(dst_cols):
            # For empty row cells
            if i % (row_spacing + 1) != 0:
                # Pixel value = round of average of right and left cells if not at last row, if at it copy last row
                resized_mat[i][j] = custom_round((resized_mat[i-1][j] + resized_mat[min(i+1, dst_rows-1)][j]) / 2) if i != dst_rows - 1 else resized_mat[i-1][j]
            # For empty column cells
            elif j % (col_spacing + 1) != 0:
                resized_mat[i][j] = custom_round((resized_mat[i][j-1] + resized_mat[i][min(j+1, dst_cols-1)]) / 2) if j != dst_cols - 1 else resized_mat[i][j-1]
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

#----------------- matrix operations ----------------------

def basic_operations(list_of_matrices: List[Matrix], operation: Literal['+', '-', '*', '/'], bit_depth: int = 8) -> Matrix:
    """
    Perform a specified operation on a list of matrices.

    Parameters:
    - `list_of_matrices` (List[Matrix]): A list of matrices on which to perform the operation.
    - `operation` (Literal['+', '-', '*', '/']): The arithmetic operation to perform on the matrices. Supported operations are addition ('+'), subtraction ('-'), multiplication ('*'), and division ('/').
    - `bit_depth` (int, optional): The bit depth for clamping the result values. Defaults to 8.

    Returns:
    - `Matrix`: A matrix that is the result of performing the specified operation on the input matrices.

    Steps:
    1. Initialize the result matrix as the first matrix in the list.
    2. Iterate through each matrix in the list, starting from the second one.
    3. For each matrix, iterate through its elements and perform the specified operation with the corresponding elements in the result matrix.
    4. Clamp the result values to the range [0, 2 ** bit_depth - 1].
    5. Handle division by zero by setting the result to 0 for such cases.

    Example:
    >>> matrix1 = [
    ...     [1, 2, 3],
    ...     [4, 5, 6],
    ...     [7, 8, 9]
    ... ]
    >>> matrix2 = [
    ...     [9, 8, 7],
    ...     [6, 5, 4],
    ...     [3, 2, 1]
    ... ]
    >>> result = basic_operations([matrix1, matrix2], '+')
    >>> for row in result:
    ...     print(row)
    [10, 10, 10]
    [10, 10, 10]
    [10, 10, 10]
    """
    if not list_of_matrices:
        return []
    max_value = 2 ** bit_depth - 1
    result = list_of_matrices[0]
    for mat in list_of_matrices[1:]:
        for i in range(len(result)):
            for j in range(len(result[0])):
                if operation == '+':
                    result[i][j] += mat[i][j]
                elif operation == '-':
                    result[i][j] -= mat[i][j]
                elif operation == '*':
                    result[i][j] *= mat[i][j]
                elif operation == '/':
                    if mat[i][j] != 0:
                        result[i][j] /= mat[i][j]
                        result[i][j] = custom_round(result[i][j])
                    else:
                        result[i][j] = 0
                result[i][j] = clip(result[i][j],0,max_value)
    return result

def logical_operations(mat1: Matrix, mat2: Matrix, operation: Literal['AND', 'OR', 'XOR'], bit_depth: int = 8) -> Matrix:
    """
    Perform logical operations on two matrices.

    Parameters:
    - `mat1` (Matrix): The first matrix.
    - `mat2` (Matrix): The second matrix.
    - `operation` (Literal['AND', 'OR', 'XOR']): The logical operation to perform. Supported operations are 'AND', 'OR', and 'XOR'.

    Returns:
    - `Matrix`: A matrix that is the result of performing the specified logical operation on the input matrices.

    Example:
    >>> mat1 = [
    ...     [1, 0, 1],
    ...     [0, 1, 0],
    ...     [1, 1, 1]
    ... ]
    >>> mat2 = [
    ...     [1, 1, 0],
    ...     [0, 0, 1],
    ...     [1, 0, 1]
    ... ]
    >>> result = logical_operations(mat1, mat2, 'AND')
    >>> for row in result:
    ...     print(row)
    [1, 0, 0]
    [0, 0, 0]
    [1, 0, 1]
    """
    max_value = 2 ** bit_depth - 1
    rows, cols = len(mat1), len(mat1[0])
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if operation == 'AND':
                result[i][j] = mat1[i][j] & mat2[i][j]
            elif operation == 'OR':
                result[i][j] = mat1[i][j] | mat2[i][j]
            elif operation == 'XOR':
                result[i][j] = mat1[i][j] ^ mat2[i][j]
            result[i][j] = clip(result[i][j],0,max_value)
    return result

def complement(matrix: Matrix, bit_depth: int=8, threshold: int=0) -> Matrix:
    """
    Complement an image and perform solarization if a threshold is given.

    Parameters:
    - `matrix` (Matrix): The input matrix representing an image.
    - `bit_depth` (int, optional): The bit depth of the image. Defaults to 8.
    - `threshold` (int, optional): The threshold value for solarization. Defaults to 0.

    Returns:
    - `Matrix`: The complemented and optionally solarized image matrix.

    Example:
    >>> image = [
    ...     [100, 200, 50],
    ...     [150, 75, 225],
    ...     [25, 175, 125]
    ... ]
    >>> complemented_image = complement(image, threshold=100)
    >>> for row in complemented_image:
    ...     print(row)
        [ 155  55  50 ]
        [ 105  75  30 ]
        [  25  80 130 ]
    """
    max_value = 2 ** bit_depth - 1
    
    # Clip the matrix values between 0 and max_value
    matrix = [[min(max(pixel, 0), max_value) for pixel in row] for row in matrix]
    
    # Apply complementing and solarization
    matrix = [[max_value - pixel if pixel >= threshold else pixel for pixel in row] for row in matrix]
    
    return matrix

def abs_diff(matrix1: Matrix, matrix2: Matrix) -> Matrix:
    """
    Calculate the absolute difference between pixel intensities of two matrices.

    Parameters:
    - `matrix1` (Matrix): The first matrix represented as a list of lists.
    - `matrix2` (Matrix): The second matrix represented as a list of lists.

    Returns:
    - `Matrix`: A matrix representing the absolute differences between corresponding elements of the input matrices.
    """

    result = []
    for row1, row2 in zip(matrix1, matrix2):
        result_row = [abs(p1 - p2) for p1, p2 in zip(row1, row2)]
        result.append(result_row)
    return result

def max_matrix(list_of_matrices: List[Matrix]) -> Matrix:
    """
    Calculate the maximum pixel intensity across all matrices.

    Parameters:
    - `list_of_matrices` (List[Matrix]): A list of matrices, where each matrix is represented as a list of lists.

    Returns:
    - `Matrix`: A matrix representing the maximum values from corresponding elements of the input matrices.
    """

    result = []
    # ALL rows from same place returned in tupels
    for pixels in zip(*list_of_matrices):
        # for each element in same place rows find the max value
        result_row = [max(pixel) for pixel in zip(*pixels)]
        result.append(result_row)
    return result

def min_matrix(list_of_matrices: List[Matrix]) -> Matrix:
    """
    Calculate the maximum pixel intensity across all matrices.

    Parameters:
    - `list_of_matrices` (List[Matrix]): A list of matrices, where each matrix is represented as a list of lists.

    Returns:
    - `Matrix`: A matrix representing the maximum values from corresponding elements of the input matrices.
    """

    result = []
    for pixels in zip(*list_of_matrices):
        result_row = [min(pixel) for pixel in zip(*pixels)]
        result.append(result_row)
    return result

def avg_matrix(list_of_matrices: List[Matrix]) -> Matrix:
    """
    Calculate the average pixel intensity across all matrices.

    Parameters:
    - `list_of_matrices` (List[Matrix]): A list of matrices, where each matrix is represented as a list of lists.

    Returns:
    - `Matrix`: A matrix representing the average values from corresponding elements of the input matrices.

    Description:
    This function iterates through each corresponding row of the matrices in `list_of_matrices`. For each set of corresponding elements, it calculates the average value (integer division) and stores it in a new matrix. The result is a matrix containing the average values from the input matrices.
    """
    result = []
    for pixels in zip(*list_of_matrices):
        result_row = [sum(pixel) // len(pixel) for pixel in zip(*pixels)]
        result.append(result_row)
    return result

def matrix_operations(list_of_matrices: List[Matrix], operation: Literal['diff', 'max', 'min', 'avg'], bit_depth: int = 8) -> Matrix:
    """Perform specified operation on a list of matrices representing images.
    
    Arguments:
    - `list_of_matrices`: List of 2D numpy arrays representing images.
    - `operation`: The operation to perform. Valid operations are 'diff', 'max', 'min', and 'avg'.
    - `bit_depth`: Number of bits used to represent the pixel values (default is 8 for binary).

    Returns:
    - result: Result of the operation.
    """
    max_value = 2 ** bit_depth - 1
    
    if operation == 'diff':
        result = abs_diff(list_of_matrices[0], list_of_matrices[1])
    elif operation == 'max':
        result = max_matrix(list_of_matrices)
    elif operation == 'min':
        result = min_matrix(list_of_matrices)
    elif operation == 'avg':
        result = avg_matrix(list_of_matrices)
        result = [[custom_round(value) for value in row] for row in result]
    # Clip result to ensure it remains within the specified bit depth range
    result = [[clip(value, 0, max_value) for value in row] for row in result]
    
    return result

def apply_operations(matrix: Matrix,
                      operations: Sequence[Literal['+', '-', '*', '/']],
                      values: Sequence[int],
                      bit_depth: int = 8):
    """
    Apply a series of operations to a matrix.

    Parameters:
    - `matrix` (Matrix): The input matrix.
    - `operations` (Sequence[Literal['+', '-', '*', '/']]): A sequence containing the operations to apply.
    - `values` (Sequence[int]): A sequence containing the values for each operation.
    - `bit_depth` (int, optional): The bit depth for the operations. Defaults to 8.

    Returns:
    - `Matrix`: The matrix after applying the specified operations.

    Example:
    >>> matrix = [
    ...     [100, 200, 50],
    ...     [150, 75, 225],
    ...     [25, 175, 125]
    ... ]
    >>> operations = ['+', '-', '*', '/']
    >>> values = [10, 20, 2, 5]
    >>> result = apply_operations(matrix, operations, values, bit_depth=8)
    >>> for row in result:
    ...     print(row)
        [ 36 51 16 ]
        [ 51 26 51 ]
        [  6 51 46 ]
    """

    if len(operations) != len(values):
        raise ValueError("The length of operations and values must be the same")

    max_value = 2 ** bit_depth - 1
    result = deepcopy(matrix)

    for op, value in zip(operations, values):
        if op == '+':
            result = [[clip(x + value,0,max_value) for x in row] for row in result]
        elif op == '-':
            result = [[clip(x - value,0,max_value) for x in row] for row in result]
        elif op == '*':
            result = [[clip(x * value,0,max_value) for x in row] for row in result]
        elif op == '/':
            if value == 0:
                raise ValueError("Division by zero is not allowed")
            result = [[clip(custom_round(x / value),0,max_value) for x in row] for row in result]
        else:
            raise ValueError(f"Unsupported operation: {op}")
    
    return result

#----------------- point operations ----------------------

def gamma(mat: Matrix, gamma_value: float = 1.0, bit_depth: int = 8) -> Matrix:
    """
    Apply gamma correction to a matrix.

    Parameters:
    - `mat` (Matrix): The input matrix.
    - `gamma_value` (float, optional): The gamma value for correction. Defaults to 1.0.
    - `bit_depth` (int, optional): The bit depth used to determine the maximum value. Defaults to 8.

    Returns:
    - `Matrix`: The matrix after applying gamma correction.

    Steps:
    1. Determine the maximum possible value (`max_value`) based on the bit depth.
    2. Normalize the pixel values in the matrix to the range [0, 1] by dividing each pixel by `max_value`.
    3. Apply gamma correction to each normalized pixel value by raising it to the power of `gamma_value`.
    4. Multiply the corrected pixel values by `max_value` and convert them to integers.
    5. Clamp the corrected pixel values to ensure they fall within the valid range [0, max_value].
    6. Return the matrix after gamma correction.

    Example:
    >>> matrix = [
    ...     [100, 200, 50],
    ...     [150, 75, 225],
    ...     [25, 175, 125]
    ... ]
    >>> gamma_corrected_matrix = gamma(matrix, gamma_value=1.5, bit_depth=8)
    >>> for row in gamma_corrected_matrix:
    ...     print(row)
        [  63 177  22 ]
        [ 115  41 211 ]
        [   8 145  88 ]
    """
    max_value = 2 ** bit_depth - 1
    # doing the operation on each value then round then clip in one line
    mat_corrected = [[clip(custom_round(max_value * ((pixel / max_value) ** gamma_value)),0,max_value) for pixel in row] for row in mat]
    return mat_corrected

def histogram_stretch(mat: Matrix, bit_depth: int = 8) -> Matrix:
    """
    Perform histogram stretching on a matrix.

    Parameters:
    - `mat` (Matrix): The input matrix.
    - `bit_depth` (int, optional): The bit depth used to determine the maximum value. Defaults to 8.

    Returns:
    - `Matrix`: The matrix after histogram stretching.

    Example:
    >>> matrix = [
    ...     [100, 200, 50],
    ...     [150, 75, 225],
    ...     [25, 175, 125]
    ... ]
    >>> stretched_matrix = histogram_stretch(matrix, bit_depth=8)
    >>> for row in stretched_matrix:
    ...     print(row)
        [  96 223  32 ]
        [ 159  64 255 ]
        [   0 191 127 ]
    """

    min_val = min(min(row) for row in mat)
    max_val = max(max(row) for row in mat)
    range_val = max_val - min_val
    max_pixel_value = 2 ** bit_depth - 1
    # doing the stretching then round
    stretched = [[custom_round((pixel - min_val) * (max_pixel_value / range_val)) for pixel in row] for row in mat]
    return stretched

def histogram_equalization(matrix: Matrix, bit_depth: int = None, show: bool = False) -> Matrix:
    """
    Perform histogram equalization on a matrix.

    Parameters:
    - `matrix` (Matrix): The input matrix.
    - `bit_depth` (int, optional): The bit depth of the matrix. If not provided, it's determined automatically. Defaults to None.
    - `show` (bool, optional): A flag indicating whether to display intermediate values during processing. Defaults to False.

    Returns:
    - `Matrix`: The matrix after histogram equalization.

    Example:
    >>> matrix = [[3,3,1,4,],
                [2,3,0,4,],
                [2,6,6,3,],
                [1,4,0,6,]]

    >>> equalized_matrix = histogram_equalization(matrix, bit_depth=3, show=True)
         R:  0 1 2 3 4 5 6 7
    hist_r:  2 2 2 4 3 0 3 0
    p(r_k):  0.125 0.125 0.125 0.25 0.188 0.0 0.188 0.0
    p(s_k):  0.12 0.25 0.38 0.62 0.81 0.81 1.0 1.0
         s:  1 2 3 4 6 6 7 7
        
    >>> display_matrices([equalized_matrix],text=["Histogram Equalization Result:"])
    Histogram Equalization Result:
    [ 4 4 2 6 ]
    [ 3 4 1 6 ]
    [ 3 7 7 4 ]
    [ 2 6 1 7 ]
    """
    if bit_depth is None:
        # Determine the bit_depth automatically
        bit_depth = bit_depth_from_mat(matrix)
    elif bit_depth <= 0:
        raise ValueError("Bit depth must be positive.")

    max_value = 2 ** bit_depth - 1

    flat = [pixel for row in matrix for pixel in row]
    total_pixels = len(flat)
    # make dict for each value in the range
    hist = {i: 0 for i in range(max_value + 1)}
    # calc the freq for each value
    for pixel in flat:
        hist[pixel] += 1

    pdf = {key: round(value / total_pixels,3) for key, value in hist.items()}
    # Calculate cumulative distribution function (CDF)
    cdf = {}
    cumulative = 0
    for key, value in pdf.items():
        cumulative += value
        cdf[key] = round( cumulative ,2)

    # For histogram equalization, map each pixel value to its equalized value
    # based on its cumulative distribution function (CDF)
    new_values = {key: custom_round(cdf[key] * max_value) for key in cdf}

    if show:
        print("     R: ", *hist.keys())
        print("hist_r: ", *hist.values())
        print("p(r_k): ", *pdf.values())
        print("p(s_k): ", *cdf.values())
        print("     s: ", *new_values.values())

    new_matrix = [[new_values[pixel] for pixel in row] for row in matrix]
    return new_matrix

def histogram_matching(matrix: Matrix, 
                       target_histogram: List[float], 
                       bit_depth: int = None, 
                       show: bool = False) -> Matrix:
    """
    Perform histogram matching on a matrix.

    Parameters:
    - `matrix` (Matrix): The input matrix.
    - `target_histogram` (List[float]): The target histogram for histogram matching.
    - `bit_depth` (int, optional): The bit depth of the matrix. If not provided, it's determined automatically. Defaults to None.
    - `show` (bool, optional): A flag indicating whether to display intermediate values during processing. Defaults to False.

    Returns:
    - `Matrix`: The matrix after histogram matching.

    Description:
    This function applies histogram matching to the input matrix, matching its histogram to a given target histogram by finding the closest values in the cumulative distribution functions. 
    It supports optional visualization of intermediate values if `show` is set to True.

    Example:
    >>> matrix = [[0,4,2,6,4],
          [4,2,4,5,6],
          [2,4,0,4,6],
          [6,6,6,0,6]]

    >>> target_histogram = [1,0,5,0,6,6,2,5]

    >>> matched_matrix = histogram_matching(matrix, target_histogram, bit_depth=3, show=True)

         R:  0 1 2 3 4 5 6 7
    p(r_k):  0.15 0.0 0.15 0.0 0.3 0.05 0.35 0.0
    p(s_k):  0.15 0.15 0.3 0.3 0.6 0.65 1.0 1.0
    p(z_k):  0.04 0.04 0.24 0.24 0.48 0.72 0.8 1.0
         s:  2 2 2 2 4 5 7 7

    >>> display_matrices([matched_matrix],text=["Histogram Matching Result:"])

    Histogram Matching Result:
    [ 2 4 2 7 4 ]
    [ 4 2 4 5 7 ]
    [ 2 4 2 4 7 ]
    [ 7 7 7 2 7 ]
    """
    if bit_depth is None:
        # Determine the bit_depth automatically
        bit_depth = bit_depth_from_mat(matrix)
    elif bit_depth <= 0:
        raise ValueError("Bit depth must be positive.")

    max_value = 2 ** bit_depth - 1

    if target_histogram is None:
        raise ValueError("Target histogram is required for histogram matching.")
    elif len(range(max_value + 1)) != len(target_histogram):
        raise ValueError("Target histogram must be same range of values of matrix.")

    flat = [pixel for row in matrix for pixel in row]
    # make dict for each value in the range
    hist = {i: 0 for i in range(max_value + 1)}
    # calc the freq for each value
    for pixel in flat:
        hist[pixel] += 1

    def _calculate_cdf(hist: Dict[int, int]) -> Dict[int, float]:
        # the total may diff from hist to another
        total_pixels = sum(hist.values())
        pdf = {key: round(value / total_pixels,3) for key, value in hist.items()}
        """Calculate the cumulative distribution function (CDF) from a histogram."""
        # Calculate cumulative distribution function (CDF)
        cdf = {}
        cumulative = 0
        for key, value in pdf.items():
            cumulative += value
            cdf[key] = round( cumulative ,2)
        return cdf,pdf

    cdf,pdf = _calculate_cdf(hist)

    # For histogram matching, calculate the target cumulative distribution function (CDF)
    # from the given target histogram and map each pixel value to the closest value
    # in the target CDF

    # make target hist key value pairs since we are taking values only
    target_hist = {i: freq for i, freq in enumerate(target_histogram)}
    # Calculate the target cumulative distribution function (CDF)
    target_cdf, _ = _calculate_cdf(target_hist)
    new_values = {}
    for key in cdf:
        # Find the closest key in the target CDF to match the CDF of the original pixel value
        diff = float('inf')
        closest_key = None
        for target_key in target_cdf:
            if abs(cdf[key] - target_cdf[target_key]) < diff:
                diff = abs(cdf[key] - target_cdf[target_key])
                closest_key = target_key
        new_values[key] = closest_key

    if show:
        # Calculate the maximum length of elements in the matrix
        max_length = max(len(str(number)) for number in pdf.values())
        R = " ".join((str(number)).ljust(max_length) for number in hist.keys())

        print("     R: ", R)
        print("p(r_k): ", *pdf.values())
        print("p(s_k): ", *cdf.values())
        print("p(z_k): ", *target_cdf.values())
        print("     s: ", *new_values.values())

    new_matrix = [[new_values[pixel] for pixel in row] for row in matrix]
    return new_matrix

# -------------------- segmentaion -------------------------

def binarizarion(matrix: Matrix, threshold: int = None, bit_depth:int = 1) -> Matrix:
    
    max_value = 2 ** bit_depth - 1

    if threshold == None:
        threshold = sum(elem for rows in matrix for elem in rows)/(len(matrix) * len(matrix[0]))

    return [[max_value if pixel > threshold else 0 for pixel in row] for row in matrix]

def dubble_threshold(matrix: Matrix, threshold1: int = 85, threshold2: float = 170, bit_depth:int = 8) -> Matrix:

    max_value = 2 ** bit_depth - 1

    return [[max_value if (threshold1 < pixel and pixel < threshold2)  else 0 for pixel in row] for row in matrix]

def automatic_threshold(matrix: Matrix, threshold: float = 0.2, bit_depth:int = 1, max_iterations: int = 10, show_steps: bool = False) -> Matrix:
    """
    Apply automatic thresholding to a matrix using the specified algorithm.

    Parameters:
    - `matrix` (Matrix): The input matrix.
    - `threshold` (float, optional): The convergence threshold for the algorithm. Defaults to 0.2.
    - `max_iterations` (int, optional): The maximum number of iterations allowed. Defaults to 10.
    - `show` (bool, optional): A flag indicating whether to display the intermediate threshold values during iterations. Defaults to False.

    Returns:
    - `Matrix`: The matrix after applying automatic thresholding.

    Description:
    This function applies automatic thresholding to each pixel value in the input matrix using the specified algorithm. The algorithm iteratively calculates the threshold value that separates the pixels into two classes, optimizing the threshold to maximize the separation.

    Steps:
    1. Initialize the old threshold (`theta_old`) with the mean value of the matrix.
    2. Calculate the mean values of pixels below and above the threshold (`mu_1` and `mu_2`).
    3. Compute the new threshold (`theta_new`) as the average of `mu_1` and `mu_2`.
    4. Check for convergence: if the absolute difference between `theta_new` and `theta_old` is less than or equal to the threshold, or the maximum number of iterations is reached, exit the loop.
    5. Update `theta_old` with `theta_new`.
    6. Increment the iteration count.
    7. Apply the final threshold to the matrix: set pixels with values greater than the threshold to 255 (white) and others to 0 (black).


    Example:
    >>> matrix = [
    ...     [100, 200, 50],
    ...     [150, 75, 225],
    ...     [25, 175, 125]
    ... ]
    >>> thresholded_matrix = automatic_threshold(matrix, threshold=0.2, max_iterations=10, show=True)
        theta_0: 125.000
        theta_1: 131.250
    >>> for row in thresholded_matrix:
    ...     print(row)
        [   0 255   0 ]
        [ 255   0 255 ]
        [   0 255   0 ]
    """

    max_value = 2 ** bit_depth - 1

    def mean(mat: Matrix) -> float:
        count = len(mat)*len(mat[0])
        total = sum(sum(row) for row in mat)
        return total / count

    def mean_below(mat: Matrix, threshold_value: float) -> float:
        values = [pixel for row in mat for pixel in row if pixel <= threshold_value]
        return sum(values) / len(values) if values else 0

    def mean_above(mat: Matrix, threshold_value: float) -> float:
        values = [pixel for row in mat for pixel in row if pixel > threshold_value]
        return sum(values) / len(values) if values else 0

    theta_old = mean(matrix)
    iteration = 0
    if show_steps:
        print(f'theta_{iteration}: {theta_old:0.3f}')
    while True:
        mu_1 = mean_below(matrix, theta_old)
        mu_2 = mean_above(matrix, theta_old)
        theta_new = (mu_1 + mu_2) / 2

        if abs(theta_new - theta_old) <= threshold or iteration >= max_iterations:
            break

        theta_old = theta_new
        iteration += 1
        if show_steps:
            print(f'theta_{iteration}: {theta_new:.3f}')

    return [[max_value if pixel > theta_new else 0 for pixel in row] for row in matrix]

