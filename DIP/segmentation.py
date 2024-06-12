from typing import Optional
from .helper import Matrix

def binarizarion(matrix: Matrix, threshold: Optional[int] = None, bit_depth:int = 1) -> Matrix:
    """
    Apply basic thresholding to an image.
    
    Parameters:
        `matrix` (Matrix): Input image matrix
        `threshold` (int, optional): Threshold value (if None, uses mean of matrix)
        `bit_depth` (int): Output bit depth
        
    Returns:
        Matrix: Binary matrix where values above threshold are max_value, others are 0
    """
    max_value = 2 ** bit_depth - 1

    if threshold is None:
        threshold = sum(elem for rows in matrix for elem in rows)/(len(matrix) * len(matrix[0]))

    return [[max_value if pixel > threshold else 0 for pixel in row] for row in matrix]

def double_threshold(
    matrix: Matrix, 
    lower_threshold: int = 85, 
    upper_threshold: int = 170, 
    bit_depth: int = 8
) -> Matrix:
    """
    Applies dual thresholding to an image matrix, producing a binary output matrix. 
    Pixel values within the threshold range are set to the maximum value based on 
    the specified bit depth, while others are set to zero.

    Parameters:
        `matrix` (Matrix): The input image matrix to process.
        `lower_threshold` (int, optional): Lower boundary for thresholding.
        `upper_threshold` (int, optional): Upper boundary for thresholding.
        `bit_depth` (int, optional): Bit depth to determine max output value.

    Returns:
        Matrix: A binary matrix where pixel values between thresholds are set to 
                `max_value` (based on `bit_depth`), and others are set to 0.

    Raises:
        ValueError: If `lower_threshold` is not less than `upper_threshold`.

    Examples:
        >>> img = [[100, 120, 180],
        ...        [60, 85, 200],
        ...        [90, 150, 160]]
        >>> processed_img = double_threshold(img, lower_threshold=85, upper_threshold=170, bit_depth=8)
        >>> # Result will highlight values in the 85-170 range.
    """
    if not 0 <= lower_threshold < upper_threshold:
        raise ValueError("Lower threshold must be less than upper threshold")
    
    max_value = 2 ** bit_depth - 1

    return [
        [max_value if lower_threshold < pixel < upper_threshold else 0
          for pixel in row]
            for row in matrix
        ]

def automatic_threshold(matrix: Matrix, threshold: float = 0.2, bit_depth:int = 1, max_iterations: int = 10, show_steps: bool = False) -> Matrix:
    """
    Apply automatic thresholding to a matrix using the specified algorithm.

    Parameters:
        `matrix` (Matrix): The input matrix.
        `threshold` (float, optional): The convergence threshold for the algorithm. Defaults to 0.2.
        `max_iterations` (int, optional): The maximum number of iterations allowed. Defaults to 10.
        `show` (bool, optional): A flag indicating whether to display the intermediate threshold values during iterations. Defaults to False.

    Returns:
        `Matrix`: The matrix after applying automatic thresholding.

    Steps:
    1. Initialize the old threshold (`theta_old`) with the mean value of the matrix.
    2. Calculate the mean values of pixels below and above the threshold (`mu_1` and `mu_2`).
    3. Compute the new threshold (`theta_new`) as the average of `mu_1` and `mu_2`.
    4. Check for convergence: if the absolute difference between `theta_new` and `theta_old` is less than or equal to the threshold, or the maximum number of iterations is reached, exit the loop.
    5. Update `theta_old` with `theta_new`.
    6. Increment the iteration count.
    7. Apply the final threshold to the matrix: set pixels with values greater than the threshold to 255 (white) and others to 0 (black).


    Examples:
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

    def mean(matrix: Matrix) -> float:
        count = len(matrix)*len(matrix[0])
        total = sum(sum(row) for row in matrix)
        return total / count

    def mean_below(matrix: Matrix, threshold_value: float) -> float:
        values = [pixel for row in matrix for pixel in row if pixel <= threshold_value]
        return sum(values) / len(values) if values else 0

    def mean_above(matrix: Matrix, threshold_value: float) -> float:
        values = [pixel for row in matrix for pixel in row if pixel > threshold_value]
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
