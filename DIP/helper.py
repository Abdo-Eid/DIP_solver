from typing import List, Tuple, Literal, Union, Sequence, Dict, Generator
from copy import deepcopy
import math as m
from random import randint
import os

Matrix = List[List[int]]

def make_random(width: int = 3, height: int = 3, bit_depth: int = 8) -> Matrix:
    """
    Create a random matrix with specified dimensions and bit depth.

    Parameters:
    - `width` (int, optional): The number of columns in the matrix. Defaults to 3.
    - `height` (int, optional): The number of rows in the matrix. Defaults to 3.
    - `bit_depth` (int, optional): The bit depth for the random values. Defaults to 8.

    Returns:
    - `Matrix`: A matrix of the specified dimensions filled with random integers based on the bit depth.

    Description:
    This function generates a random matrix with the given `width`, `height`, and `bit_depth`. The `bit_depth` determines the range of random values, where the maximum value is `2 ** bit_depth - 1`.
    """

    max_value = 2 ** bit_depth
    matrix = [[randint(0, max_value - 1) for _ in range(width)] for _ in range(height)]
    return matrix

def get_center(width: int = 3, height: int = 3, return_matrix: bool = False) -> Union[Matrix, Tuple[int, int]]:
    """
    Get the center coordinate of a matrix of the specified shape, optionally displaying it.

    Parameters:
    - `width` (int, optional): The number of columns in the matrix. Defaults to 3.
    - `height` (int, optional): The number of rows in the matrix. Defaults to 3.
    - `return_matrix` (bool, optional): A boolean indicating whether to return the matrix with the center coordinate marked. Defaults to False.

    Returns:
    - If `return_matrix` is True, returns a matrix of '*' with the center coordinate marked as "X".
    - If `return_matrix` is False, returns a tuple containing the row and column indices of the center coordinate.

    Example:
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
    - `matrix` (Matrix): The matrix from which to retrieve the pixel value. A matrix is a list of lists of integers.
    - `index` (Tuple[int, int]): A tuple specifying the row and column indices of the pixel.

    Returns:
    - `int`: The value of the pixel at the specified index.

    Example:
    >>> matrix = [[1, 2, 3],
    ...           [4, 5, 6],
    ...           [7, 8, 9]]
    >>> get_pixel(matrix, (1, 2))
        6
    """
    return matrix[index[0]][index[1]]

def display_matrices(list_of_matrices: List[Matrix], text: List[str] = [], coordinates: bool = False) -> None:
    """
    Display a list of matrices with optional coordinate labels.

    Parameters:
    - `list_of_matrices` (List[Matrix]): A list of matrices to be displayed. Each matrix is a list of lists.
    - `coordinates` (bool, optional): If True, display row and column numbers. Defaults to False.

    Returns:
    - None

    Description:
    This function prints each matrix in the list_of_matrices. If the coordinates parameter is set to True, it prints row and column numbers along with the matrix. Each matrix is displayed with elements right-aligned for better readability.

    Example:
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
    text_flag = False
    if len(list_of_matrices) == 1 and (text == []):
        text = [""]
        text_flag = True
    elif text == []:
        pass
    elif (len(text) != len(list_of_matrices)):
        raise ValueError("The length of list_of_matrices and text must be the same")
    else:
        text_flag = True

    # Loop over each matrix in the list_of_matrices
    for i, matrix in enumerate(list_of_matrices):
        # Print the matrix number
        if text_flag:
            print(text[i])
        else:
            print(f"Matrix {i + 1}:")
        
        # Calculate the maximum length of elements in the matrix
        max_length = max(len(str(item)) for row in matrix for item in row)
        

        # Print column numbers if coordinates is True
        if coordinates:
            print("    ", end="")
            for col_num in range(len(matrix[0])):
                print(str(col_num).rjust(max_length), end=" ")
            print()
        # Print each row of the matrix with appropriate spacing and brackets
        for idx, row in enumerate(matrix):
            # Print row number if coordinates is True
            if coordinates:
                print(f"{idx}", end=" ")
            # Join elements of the row with appropriate spacing and right-align each element
            print("[ " + " ".join((str(item)).rjust(max_length) for item in row) + " ]")
        
        # Print an empty line to separate matrices
        print()

def custom_round(x: float) -> int:
    """
    Round a floating-point number to the nearest integer using custom rounding logic.

    Parameters:
    - `x` (float): The floating-point number to be rounded.

    Returns:
    - `int`: The integer result of rounding `x` to the nearest integer.

    Description:
    This function rounds a floating-point number `x` to the nearest integer. It adds 0.5 to `x` and then converts the result to an integer. This custom rounding logic ensures that numbers with a fractional part of 0.5 or higher are rounded up, while those with a fractional part less than 0.5 are rounded down.

    Example:
    >>> custom_round(2.3)
    2
    >>> custom_round(2.5)
    3
    >>> custom_round(2.8)
    3
    """

    return int(x + 0.5)

def clip(x:int,min_value:int, max_value:int)->int:
    return min(max(x, min_value), max_value)

def mat_max(mat:Matrix) -> int:
    """
    Find the maximum value in a given matrix.

    Parameters:
    - `mat` (Matrix): A matrix represented as a list of lists, where each inner list represents a row.

    Returns:
    - `int`: The maximum value found in the matrix.
    """

    return max(max(row) for row in mat)

def bit_depth_from_mat(mat:Matrix)->int:
    return m.ceil(m.log2(mat_max(mat) + 1))

def make_question(original:Matrix, result:Matrix):
    """
    1. will show the original image with a text that askes about a pixel in the result image
    2. ask for the answer
    3. if right exit if wrong try again
    4. 'quit' to quit, '?' for the answer
    """
    result_width = len(result[0])
    result_hieght = len(result)

    pixel = randint(0, result_hieght - 1),randint(0, result_width - 1)
    result_answer = str(get_pixel(result,pixel))


    os.system('cls' if os.name == 'nt' else 'clear')

    wrong = False
    while(True):

        display_matrices([original],[f'what is the value of pixel {pixel}: '], coordinates= True)
        if wrong:
            print('wrong input\n')
        answer = input("Enter '?' for the answer.\nEnter 'quit' to exit.\nthe answer is: ").lower()
        if answer == result_answer:
            print("Correct")
            break
        elif answer == '?':
            print('The answer is', result_answer)
            break
        elif answer == 'quit':
            print('bye')
            break
        else:
            wrong = True
            os.system('cls' if os.name == 'nt' else 'clear')

def plot_morphology(images, titles):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for ax, img, title in zip(axes, images, titles):
        ax.matshow(img, cmap='gray_r')
        ax.set_title(title)
        ax.axis('off')

        # Add border around each element
        num_rows, num_cols = len(img),len(img[0])
        for i in range(num_rows):
            for j in range(num_cols):
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', linewidth=1)
                ax.add_patch(rect)

    plt.show()

