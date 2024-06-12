from typing import Literal, Sequence, Dict, Tuple, List
from copy import deepcopy
from .helper import Matrix, normal_round, clip, bit_depth_from_mat

#----------------- matrix operations ----------------------

def mat_operate(list_of_matrices: List[Matrix],
                      operation: Literal['+', '-', '*', '/'],
                        bit_depth: int = 8) -> Matrix:
    """
    Performs a specified arithmetic operation on a list of matrices.

    Parameters:
        `list_of_matrices` (List[Matrix]): A list of matrices on which to perform the operation.
        `operation` (Literal['+', '-', '*', '/']): The arithmetic operation to perform. Supported operations are addition ('+'), subtraction ('-'), multiplication ('*'), and division ('/').
        `bit_depth` (int, optional): The bit depth for clamping result values. Defaults to 8.

    Returns:
        Matrix: The resulting matrix after performing the specified operation.

    Examples:
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
    result = deepcopy(list_of_matrices[0])

    operations = {'+':lambda x, y : x+y,
                  '-':lambda x, y : x-y,
                  '*':lambda x, y : x*y,
                  "/":lambda x, y : normal_round(x/y) if y != 0 else 0}

    for matrix in list_of_matrices[1:]:
        for i,_ in enumerate(result):
            for j in range(len(result[0])):
                result[i][j] = clip(operations[operation](result[i][j],matrix[i][j]),0,max_value)
    return result

def mat_logical_operate(matrix1: Matrix, matrix2: Matrix,
                        operation: Literal['AND', 'OR', 'XOR'],
                          bit_depth: int = 8) -> Matrix:
    """
    Perform logical operations on two matrices.

    Parameters:
        `matrix1` (Matrix): The first matrix.
        `matrix2` (Matrix): The second matrix.
        `operation` (Literal['AND', 'OR', 'XOR']): The logical operation to perform. Supported operations are 'AND', 'OR', and 'XOR'.

    Examples:
    >>> matrix1 = [
    ...     [1, 0, 1],
    ...     [0, 1, 0],
    ...     [1, 1, 1]
    ... ]
    >>> matrix2 = [
    ...     [1, 1, 0],
    ...     [0, 0, 1],
    ...     [1, 0, 1]
    ... ]
    >>> result = logical_operations(matrix1, matrix2, 'AND')
    >>> for row in result:
    ...     print(row)
    ... [1, 0, 0]
    ... [0, 0, 0]
    ... [1, 0, 1]
    """
    max_value = 2 ** bit_depth - 1
    rows, cols = len(matrix1), len(matrix1[0])
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    operations = {'AND':lambda x, y : x&y,
                  'OR':lambda x, y : x|y,
                  'XOR':lambda x, y : x^y}
    for i in range(rows):
        for j in range(cols):
            result[i][j] = clip(operations[operation](matrix1[i][j],matrix2[i][j]), 0, max_value)
    return result

def complement(matrix: Matrix, bit_depth: int=8, threshold: int=0) -> Matrix:
    """
    Complement an image and perform solarization if a threshold is given.

    Parameters:
        `matrix` (Matrix): The input matrix representing an image.
        `bit_depth` (int, optional): The bit depth of the image. Defaults to 8.
        `threshold` (int, optional): The threshold value for solarization. Defaults to 0.

    Returns:
        Matrix: The complemented and optionally solarized image matrix.

    Examples:
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

def _abs_diff(matrix1: Matrix, matrix2: Matrix) -> Matrix:
    """
    Calculate the absolute difference between pixel intensities of two matrices.

    Parameters:
        `matrix1` (Matrix): The first matrix represented as a list of lists.
        `matrix2` (Matrix): The second matrix represented as a list of lists.

    Returns:
        Matrix: A matrix representing the absolute differences between corresponding elements of the input matrices.

    Examples:
    >>> matrix_a = [[255 ,100],
                    [50 ,200]]
    >>> matrix_b = [[100 ,150],
                    [200 ,50]]
    >>> abs_difference_result = abs_diff(matrix_a ,matrix_b)
    >>> for row in abs_difference_result:
    ...     print(row)
    [155 ,50]
    [150 ,150]
    """

    result = []
    for row1, row2 in zip(matrix1, matrix2):
        result_row = [abs(p1 - p2) for p1, p2 in zip(row1, row2)]
        result.append(result_row)
    return result

def _matrices_max(list_of_matrices: List[Matrix]) -> Matrix:
    """
    Calculate the maximum pixel intensity across all matrices.

    Parameters:
        `list_of_matrices` (List[Matrix]): A list of matrices, where each matrix is represented as a list of lists.

    Returns:
        Matrix: A matrix representing the maximum values from corresponding elements of the input matrices.
        Examples:
        >>> matrix1 = [[10 ,20],
                        [30 ,40]]
        >>> matrix2 = [[15 ,25],
                        [35 ,45]]
        >>> max_result = matrices_max([matrix1,matrix2])
        >>> for row in max_result:
        ...     print(row)
        [15 ,25]
        [35 ,45]
    """

    result = []
    # ALL rows from same place returned in tupels
    for pixels in zip(*list_of_matrices):
        # for each element in same place rows find the max value
        result_row = [max(pixel) for pixel in zip(*pixels)]
        result.append(result_row)
    return result

def _matrices_min(list_of_matrices: List[Matrix]) -> Matrix:
    """
    Calculate the maximum pixel intensity across all matrices.

    Parameters:
        `list_of_matrices` (List[Matrix]): A list of matrices, where each matrix is represented as a list of lists.

    Returns:
        Matrix: A matrix representing the maximum values from corresponding elements of the input matrices.

    Examples:
    >>> matrix1 = [[10 ,20],
                    [30 ,40]]
    >>> matrix2 = [[20 ,30],
                    [5 ,50]]
    >>> max_result = matrices_min([matrix1,matrix2])
    >>> for row in max_result:
    ...     print(row)
    [10 ,20]
    [5 ,40]
    """

    result = []
    for pixels in zip(*list_of_matrices):
        result_row = [min(pixel) for pixel in zip(*pixels)]
        result.append(result_row)
    return result

def _matrices_avg(list_of_matrices: List[Matrix]) -> Matrix:
    """
    Calculate the average pixel intensity across all matrices.

    Parameters:
        `list_of_matrices` (List[Matrix]): A list of matrices, where each matrix is represented as a list of lists.

    Returns:
        Matrix: A matrix representing the average values from corresponding elements of the input matrices.

    Example
    >>> matrix1 = [[10 ,20],
                [30 ,40]]
    >>> matrix2 = [[20 ,30],
                [5 ,50]]
    >>> max_result = matrices_avg([matrix1,matrix2])
    >>> for row in max_result:
    ...     print(row)
    [15 ,25]
    [18 ,45]
    """
    result = []
    for pixels in zip(*list_of_matrices):
        result_row = [normal_round(sum(pixel) / len(pixel)) for pixel in zip(*pixels)]
        result.append(result_row)
    return result

def aggregate_matrices(list_of_matrices: List[Matrix], operation: Literal['diff', 'max', 'min', 'avg'], bit_depth: int = 8) -> Matrix:
    """Perform specified operation on a list of matrices representing images.
    
    Arguments:
        `list_of_matrices` (List[Matrix]): List of 2D numpy arrays representing images.
        `operation` (Literal['diff', 'max', 'min', 'avg']): The operation to perform. Valid operations are 'diff', 'max', 'min', and 'avg'.
        `bit_depth` (int): Number of bits used to represent the pixel values (default is 8 for binary).

    Returns:
        Matrix: Result of the operation.

    Examples:
    >>> matrix1 = [[10 ,20],
                [30 ,40]]
    >>> matrix2 = [[20 ,30],
                [5 ,50]]
    >>> max_result = matrices_min([matrix1,matrix2], operation = 'avg', bit_depth = 4)
    >>> for row in max_result:
    ...     print(row)
    ... [15 ,15]
    ... [15 ,15]
    """
    max_value = 2 ** bit_depth - 1

    if operation == 'diff':
        result = _abs_diff(list_of_matrices[0], list_of_matrices[1])
    elif operation == 'max':
        result = _matrices_max(list_of_matrices)
    elif operation == 'min':
        result = _matrices_min(list_of_matrices)
    elif operation == 'avg':
        result = _matrices_avg(list_of_matrices)
    # Clip result to ensure it remains within the specified bit depth range
    result = [[clip(value, 0, max_value) for value in row] for row in result]

    return result

def apply_operations(matrix: Matrix,
                      operations: Sequence[Literal['+', '-', '*', '/']],
                      values: Sequence[int],
                      bit_depth: int = 8) -> Matrix:
    """
    Apply a series of operations to a matrix.

    Parameters:
        `matrix` (Matrix): The input matrix.
        `operations` (Sequence[Literal['+', '-', '*', '/']]): A sequence containing the operations to apply.
        `values` (Sequence[int]): A sequence containing the values for each operation.
        `bit_depth` (int, optional): The bit depth for the operations. Defaults to 8.

    Returns:
        Matrix: The matrix after applying the specified operations.

    Examples:
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
            result = [[clip(normal_round(x / value),0,max_value) for x in row] for row in result]
        else:
            raise ValueError(f"Unsupported operation: {op}")

    return result

#----------------- point operations ----------------------

def gamma(matrix: Matrix, gamma_value: float = 1.0, bit_depth: int = 8) -> Matrix:
    """
    Apply gamma correction to a matrix.

    Parameters:
        `matrix` (Matrix): The input matrix.
        `gamma_value` (float, optional): The gamma value for correction. Defaults to 1.0.
        `bit_depth` (int, optional): The bit depth used to determine the maximum value. Defaults to 8.

    Returns:
        Matrix: The matrix after applying gamma correction.

    Steps:
    1. Determine the maximum possible value (`max_value`) based on the bit depth.
    2. Normalize the pixel values in the matrix to the range [0, 1] by dividing each pixel by `max_value`.
    3. Apply gamma correction to each normalized pixel value by raising it to the power of `gamma_value`.
    4. Multiply the corrected pixel values by `max_value` and convert them to integers.
    5. Clamp the corrected pixel values to ensure they fall within the valid range [0, max_value].
    6. Return the matrix after gamma correction.

    Examples:
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
    mat_corrected = [[clip(normal_round(max_value * ((pixel / max_value) ** gamma_value)),0,max_value) for pixel in row] for row in matrix]
    return mat_corrected

def histogram_stretch(matrix: Matrix, bit_depth: int = 8) -> Matrix:
    """
    Perform histogram stretching on a matrix.

    Parameters:
        `matrix` (Matrix): The input matrix.
        `bit_depth` (int, optional): The bit depth used to determine the maximum value. Defaults to 8.

    Returns:
        Matrix: The matrix after histogram stretching.

    Examples:
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

    min_val = min(min(row) for row in matrix)
    max_val = max(max(row) for row in matrix)
    range_val = max_val - min_val
    max_pixel_value = 2 ** bit_depth - 1
    # doing the stretching then round
    stretched = [[normal_round((pixel - min_val) * (max_pixel_value / range_val)) for pixel in row] for row in matrix]
    return stretched

def histogram_equalization(matrix: Matrix, bit_depth: int = None, show: bool = False) -> Matrix:
    """
    Perform histogram equalization on a matrix.

    Parameters:
        `matrix` (Matrix): The input matrix.
        `bit_depth` (int, optional): The bit depth of the matrix. If not provided, it's determined automatically. Defaults to None.
        `show` (bool, optional): A flag indicating whether to display intermediate values during processing. Defaults to False.

    Returns:
        Matrix: The matrix after histogram equalization.

    Examples:
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
    ... Histogram Equalization Result:
    ... [ 4 4 2 6 ]
    ... [ 3 4 1 6 ]
    ... [ 3 7 7 4 ]
    ... [ 2 6 1 7 ]
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
    new_values = {key: normal_round(value * max_value) for key,value in cdf.items()}

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
        `matrix` (Matrix): The input matrix.
        `target_histogram` (List[float]): The target histogram for histogram matching.
        `bit_depth` (int, optional): The bit depth of the matrix. If not provided, it's determined automatically. Defaults to None.
        `show` (bool, optional): A flag indicating whether to display intermediate values during processing. Defaults to False.

    Returns:
        Matrix: The matrix after histogram matching.

    Description:
    This function applies histogram matching to the input matrix, matching its histogram to a given target histogram by finding the closest values in the cumulative distribution functions. 
    It supports optional visualization of intermediate values if `show` is set to True.

    Examples:
    >>> matrix = [[0,4,2,6,4],
          [4,2,4,5,6],
          [2,4,0,4,6],
          [6,6,6,0,6]]

    >>> target_histogram = [1,0,5,0,6,6,2,5]

    >>> matched_matrix = histogram_matching(matrix, target_histogram, bit_depth=3, show=True)
    ...      R:  0 1 2 3 4 5 6 7
    ... p(r_k):  0.15 0.0 0.15 0.0 0.3 0.05 0.35 0.0
    ... p(s_k):  0.15 0.15 0.3 0.3 0.6 0.65 1.0 1.0
    ... p(z_k):  0.04 0.04 0.24 0.24 0.48 0.72 0.8 1.0
    ...      s:  2 2 2 2 4 5 7 7

    >>> display_matrices([matched_matrix],text=["Histogram Matching Result:"])
    ... Histogram Matching Result:
    ... [ 2 4 2 7 4 ]
    ... [ 4 2 4 5 7 ]
    ... [ 2 4 2 4 7 ]
    ... [ 7 7 7 2 7 ]
    """
    if bit_depth is None:
        # Determine the bit_depth automatically
        bit_depth = bit_depth_from_mat(matrix)
    elif bit_depth <= 0:
        raise ValueError("Bit depth must be positive.")

    max_value = 2 ** bit_depth - 1

    if target_histogram is None:
        raise ValueError("Target histogram is required for histogram matching.")
    if len(range(max_value + 1)) != len(target_histogram):
        raise ValueError("Target histogram must be same range of values of matrix.")

    flat = [pixel for row in matrix for pixel in row]
    # make dict for each value in the range
    hist = {i: 0 for i in range(max_value + 1)}
    # calc the freq for each value
    for pixel in flat:
        hist[pixel] += 1

    def _calculate_cdf(hist: Dict[int, int]) -> Tuple[Dict[int, int],Dict[int, float]]:
        # the total may diff from hist to another
        total_pixels = sum(hist.values())
        pdf = {key: round(value / total_pixels,3) for key, value in hist.items()}
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
    for key,value in cdf.items():
        # Find the closest key in the target CDF to match the CDF of the original pixel value
        diff = float('inf')
        closest_key = None
        for target_key,target_value in target_cdf.items():
            if abs(value - target_value) < diff:
                diff = abs(value - target_value)
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
