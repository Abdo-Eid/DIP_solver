import numpy as np
from typing import List, Literal, Tuple, Sequence


def make_random(width: int = 3,hight: int = 3, bit_depth: int = 8):
    """making random matrix given width and hight and bit depth
    - high for rows and width for columns"""
    return np.random.randint(0, 2**bit_depth, size=(hight, width))

def get_pixel(matrix : np.ndarray,index: Tuple[int,int]):
    print(matrix[*index])

def display_matrices(list_of_matrices: List[np.ndarray]) -> None:
    """Display the matrices in the list_of_matrices.
    
    Arguments:
    - list_of_matrices: List of 2D numpy arrays representing matrices.
    """
    for i, matrix in enumerate(list_of_matrices):
        print(f"Matrix {i + 1}:\n{matrix}\n")

def resize(mat: np.ndarray, size: tuple, bit_depth: int = 8) -> np.ndarray:
    """Resize a matrix to match the size of another matrix.
    
    Arguments:
    - mat: 2D numpy array representing the matrix to be resized.
    - size: Shape of the target matrix as a tuple (rows, cols).
    - bit_depth: Number of bits used to represent the pixel values (default is 8 for binary).

    Returns:
    - resized_mat: Resized 2D numpy array.
    """
    max_value = 2 ** bit_depth - 1
    resized_mat = np.zeros(size, dtype=mat.dtype)
    rows, cols = min(mat.shape[0], size[0]), min(mat.shape[1], size[1])
    resized_mat[:rows, :cols] = np.clip(mat[:rows, :cols], 0, max_value)
    return resized_mat

custom_round_vec = np.vectorize(lambda x : int(x + 0.5) if x % 1 == 0.5 else np.round(x))

def basic_operations(list_of_matrices: List[np.ndarray], operation: Literal['+', '-', '*', '/'], bit_depth: int = 8) -> np.ndarray:
    """Perform specified operation on a list of matrices.
    
    Arguments:
    - list_of_matrices: List of 2D numpy arrays representing matrices.
    - operation: The operation to perform. Valid operations are '+', '-', '*', and '/'.
    - bit_depth: Number of bits used to represent the pixel values (default is 8 for binary).

    Returns:
    - result_matrix: Result of the operation as a 2D numpy array.
    """
    max_value = 2 ** bit_depth - 1
    matrices = [np.clip(matrix, 0, max_value) for matrix in list_of_matrices]
    
    result = matrices[0]
    if operation == '+':
        for matrix in matrices[1:]:
            result += matrix
    elif operation == '-':
        for matrix in matrices[1:]:
            result -= matrix
    elif operation == '*':
        for matrix in matrices[1:]:
            result *= matrix
    elif operation == '/':
        result = matrices[0].astype('float64')
        for matrix in matrices[1:]:
            # Perform element-wise division
            result = np.divide(result, matrix, out=np.zeros_like(result), where=matrix!=0)
    
    # Round the result matrix and clip values
    result = custom_round_vec(result).astype(matrices[0].dtype)
    result = np.clip(result, 0, max_value)
    
    return result

def logical_operations(list_of_matrices: List[np.ndarray], operation: Literal['or', 'and', 'xor'], bit_depth: int = 8) -> np.ndarray:
    """Perform specified logical operation on a list of matrices.
    
    Arguments:
    - list_of_matrices: List of 2D numpy arrays representing matrices.
    - operation: The logical operation to perform. Valid operations are 'or', 'and', and 'xor'.
    - bit_depth: Number of bits used to represent the pixel values (default is 8 for binary).

    Returns:
    - result_matrix: Result of the logical operation as a 2D numpy array.
    """
    max_value = 2 ** bit_depth - 1
    matrices = [matrix.astype(np.int64) for matrix in list_of_matrices]
    
    if operation == 'or':
        result = np.bitwise_or.reduce(matrices)
    elif operation == 'and':
        result = np.bitwise_and.reduce(matrices)
    elif operation == 'xor':
        result = np.bitwise_xor.reduce(matrices)
    
    return np.clip(result, 0, max_value)

def image_operations(list_of_matrices: List[np.ndarray], operation: Literal['diff', 'max', 'min', 'avg'], bit_depth: int = 8) -> np.ndarray:
    """Perform specified operation on a list of matrices representing images.
    
    Arguments:
    - list_of_matrices: List of 2D numpy arrays representing images.
    - operation: The operation to perform. Valid operations are 'diff', 'max', 'min', and 'avg'.
    - bit_depth: Number of bits used to represent the pixel values (default is 8 for binary).

    Returns:
    - result: Result of the operation.
    """
    max_value = 2 ** bit_depth - 1
    
    if operation == 'diff':
        # Calculate absolute difference between pixel intensities of the two images
        result = np.abs(list_of_matrices[0] - list_of_matrices[1])
    elif operation == 'max':
        # Calculate maximum pixel intensity across all images
        result = np.max(list_of_matrices, axis=0)
    elif operation == 'min':
        # Calculate minimum pixel intensity across all images
        result = np.min(list_of_matrices, axis=0)
    elif operation == 'avg':
        # Calculate average pixel intensity across all images
        result = np.mean(list_of_matrices, axis=0)
        # Vectorize the custom rounding function and apply it to the result
        result = custom_round_vec(result).astype(list_of_matrices[0].dtype)
    # Clip result to ensure it remains within the specified bit depth range
    result = np.clip(result, 0, max_value)
    
    return result

# -----------point operation -------------
def complement(matrix : np.ndarray, bit_depth: int = 8, thrushold = 0) -> np.ndarray:
    """complementing an image and solarization if thrushold is given
    
    Arguments:
    - matrix: 2D numpy arrays.
    Return:2D matrix
    """
    max_value = 2 ** bit_depth - 1
    matrix = np.clip(matrix, 0, max_value)

        
    custom_complement_vec = np.vectorize(lambda x : (max_value - x) if x >= thrushold else x)

    return custom_complement_vec(matrix)

def apply_operations(matrix : np.ndarray,
                      operations: Sequence[Literal['+', '-', '*', '/']],
                        values: Sequence[int],
                          bit_depth: int = 8) -> np.ndarray:
    """
    Apply a series of operations to a NumPy matrix.

    Parameters:
        matrix (np.ndarray): The input matrix.
        operations (Sequence of Literal['+', '-', '*', '/']): A sequence containing the operations.
        values (Sequence of int): A sequence containing the values for each operation.
        bit_depth (int, optional): Bit depth for the operations, default is 8.

    Returns:
        np.ndarray: The matrix after applying the operations.
    """
    if len(operations) != len(values):
        raise ValueError("The length of operations and values must be the same")

    max_value = 2 ** bit_depth - 1
    result = matrix.copy()

    for op, value in zip(operations, values):
        if op == '+':
            result += value
        elif op == '-':
            result -= value
        elif op == '*':
            result *= value
        elif op == '/':
            if value == 0:
                raise ValueError("Division by zero is not allowed")
            result = result.astype('float64')
            result /= value
        else:
            raise ValueError(f"Unsupported operation: {op}")
        # Round the result matrix and clip values
        result = custom_round_vec(result)
        result = np.clip(result, 0, max_value)
    
    return result.astype(matrix.dtype)

def gamma(matrix: np.ndarray, c: float = 1.0, bit_depth: int = 8) -> np.ndarray:
    """
    Apply gamma correction to a NumPy matrix.

    Gamma correction is used to adjust the brightness of an image. A gamma coefficient greater than 1 makes the image lighter, 
    while a gamma coefficient less than 1 makes it darker.

    Parameters:
        matrix (np.ndarray): The input matrix.
        c (float, optional): The gamma coefficient, ranging from 0 to infinity. Default is 1.0.
        bit_depth (int, optional): The bit depth of the matrix values. Default is 8.

    Returns:
        np.ndarray: The matrix after applying the gamma correction.
    
    Example:
        >>> matrix = np.array([[100, 150], [200, 250]], dtype=np.uint8)
        >>> gamma(matrix, c=2.0)
        array([[ 39,  88],
               [157, 244]], dtype=uint8)
    """
    result = matrix.copy().astype('float64')
    max_value = 2 ** bit_depth - 1
    result = max_value * ((result / max_value) ** c)
    result = custom_round_vec(result).astype(matrix.dtype)
    result = np.clip(result, 0, max_value)
    return result

def histogram_stretch(matrix: np.ndarray, bit_depth: int = 8, plot: bool = False) -> np.ndarray:
    """
    Apply histogram stretching to a NumPy matrix.

    Histogram stretching (also known as contrast stretching) improves the contrast of an image by stretching the range 
    of intensity values to span the full range of possible values.

    Parameters:
        matrix (np.ndarray): The input matrix.
        bit_depth (int, optional): The bit depth of the matrix values. Default is 8.
        plot (bool, optional): Whether to plot the old and new histograms. Default is False.

    Returns:
        np.ndarray: The matrix after applying histogram stretching.
    
    Example:
        >>> matrix = np.array([[100, 150], [200, 250]], dtype=np.uint8)
        >>> histogram_stretch(matrix)
        array([[  0,  85],
               [170, 255]], dtype=uint8)
    """
    result = matrix.copy().astype('float64')
    low = result.min()
    high = result.max()

    max_value = 2 ** bit_depth - 1
    result = max_value * ((result - low) / (high - low))
    result = np.clip(result, 0, max_value).astype(matrix.dtype)

    if plot:
        import matplotlib.pyplot as plt
        # Calculate old histogram
        old_hist, _ = np.histogram(matrix.flatten(), bins=np.arange(max_value+1))

        # Calculate new histogram
        new_hist, _ = np.histogram(result.flatten(), bins=np.arange(max_value+1))

        # Plot old and new histograms
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(np.arange(max_value), old_hist, width=0.8, color='b', alpha=0.5)
        plt.title('Old Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

        plt.subplot(1, 2, 2)
        plt.bar(np.arange(max_value), new_hist, width=0.8, color='r', alpha=0.5)
        plt.title('New Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    return result

def histogram_equalization(matrix: np.ndarray, bit_depth: int = None, show: bool = False, plot: bool = False) -> np.ndarray:
    """
    Apply histogram equalization to a grayscale image represented as a NumPy matrix.

    Histogram equalization enhances the contrast of an image by redistributing pixel intensities.

    Parameters:
        matrix (np.ndarray): The input matrix representing the grayscale image.
        bit_depth (int, optional): The bit depth of the image (number of intensity levels).
            If not provided, it will be calculated automatically based on the maximum value in the matrix.
        show (bool, optional): Whether to display intermediate results including histograms. Default is False.
        plot (bool, optional): Whether to plt histogram. Default is False.

    Returns:
        np.ndarray: The matrix after applying histogram equalization.

    Raises:
        ValueError: If the bit depth is not positive.
    """
    if bit_depth == None:
        bit_depth = np.ceil(np.log2(matrix.max()+1))
    elif bit_depth <= 0:
        raise ValueError("Bit depth must be positive.")

    max_value = 2 ** bit_depth - 1
    values = np.arange(max_value + 1)

    # Calculate histogram
    hist, bins = np.histogram(matrix.flatten(), bins=np.arange(max_value + 2))

    # Normalize the histogram to get probability density function (PDF)
    pdf = hist / matrix.size

    # Calculate cumulative distribution function (CDF)
    cdf = pdf.cumsum()

    # Interpolation for intensity mapping
    mapped = np.round(cdf * max_value).astype(np.uint8)

    if show:
        print("     r: ", values)
        print("hist_r: ", hist)
        print("p(r_k): ", pdf)
        print("p(s_k): ", cdf)
        print("     s: ", mapped)

    if plot:
        # Plot the histogram
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.bar(values, hist, width=0.8, color='b', alpha=0.5)
        plt.title('Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
    # Apply the lookup table to the matrix
    mapped_matrix = mapped[matrix]

    return mapped_matrix

def automatic_threshold(matrix: np.ndarray, threshold=0.2, max_iterations=10, show = False) -> np.ndarray:
    """
    Apply automatic thresholding to a NumPy matrix using the specified algorithm.

    This algorithm iteratively updates the threshold until convergence or reaching the maximum number of iterations.

    Parameters:
        matrix (np.ndarray): The input matrix.
        threshold (float, optional): The convergence threshold. Default is 0.2.
        max_iterations (int, optional): The maximum number of iterations. Default is 10.

    Returns:
        np.ndarray: The matrix after applying the automatic thresholding.
    """
    theta_old = np.mean(matrix)
    iteration = 0

    # Step 1: Calculate theta_0
    theta_old = np.mean(matrix)
    print(f'theta_{iteration}: {theta_old:0.3f}')
    while True:
        # Step 2: Calculate Mu_1 and Mu_2
        Mu_1 = np.mean(matrix[matrix <= theta_old])
        Mu_2 = np.mean(matrix[matrix > theta_old])

        # Step 3: Calculate new threshold
        theta_new = (Mu_1 + Mu_2) / 2

        # Step 4: Check convergence or maximum iterations
        if np.abs(theta_new - theta_old) <= threshold or iteration >= max_iterations:
            break

        theta_old = theta_new
        iteration += 1
        if show:
            print(f'theta_{iteration}: {theta_new:0.3f}')

    return (matrix > theta_new).astype(np.uint8) * 255