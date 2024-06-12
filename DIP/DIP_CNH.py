import random
from typing import List, Union, Tuple

Matrix = List[List[int]]

class MatrixImage:
    def __init__(self, matrix:Matrix, bit_depth: int = 8):
        # Check if all rows have the same length
        row_lengths = [len(row) for row in matrix]
        if len(set(row_lengths)) != 1:
            raise ValueError("Matrix must be rectangular.")
        self.matrix = matrix
        self.width = len(matrix[0])
        self.height = len(matrix)
        self.dim = (self.width,self.height)
        self.size = self.width*self.height
        self.bit_depth = bit_depth
        self.max_value = 2 ** bit_depth - 1
        self.center = self.get_center()
        
    @classmethod
    def make_random(cls, width: int = 3, height: int = 3, bit_depth: int = 8):
        """
        Create a random matrix with specified dimensions and bit depth.

        Parameters:
        - `width` (int, optional): The number of columns in the matrix. Defaults to 3.
        - `height` (int, optional): The number of rows in the matrix. Defaults to 3.
        - `bit_depth` (int, optional): The bit depth for the random values. Defaults to 8.

        Returns:
        - `Matrix`: A matrix of the specified dimensions filled with random integers based on the bit depth.
        """

        max_value = 2 ** bit_depth - 1
        matrix = [[random.randint(0, max_value) for _ in range(width)] for _ in range(height)]
        return cls(matrix)
    @classmethod
    def get_center(cls, width: int = 3, height: int = 3, return_matrix: bool = False) -> Union[Matrix, Tuple[int, int]]:
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

    def get_center(self, show_center: bool = False) -> Union[Matrix, Tuple[int, int]]:
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
        x,y = self.height//2, self.width//2

        if show_center:
            self.display_matrices([[["*" if row != x or col != y else "X" for col in range(self.width)] for row in range(self.height)]],[f"Center: {x},{y}"],coordinates=True) 
        else:
            return x,y
    @classmethod
    def display_matrices(cls, list_of_matrices: List[Matrix], text: List[str], coordinates: bool = False) -> None:
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
        if len(text) != len(list_of_matrices):
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

    def __repr__(self) -> str:

        # Calculate the maximum length of elements in the matrix
        max_length = max(len(str(item)) for row in self.matrix for item in row)
        
        text = ""

        # Print each row of the matrix with appropriate spacing and brackets
        for row in self.matrix:
            # Join elements of the row with appropriate spacing and right-align each element
            text += "[ " + " ".join((str(item)).rjust(max_length) for item in row) + " ]\n"
        return text
        
class Kernal(MatrixImage):
    def __init__(self, matrix: Matrix, bit_depth: int = 8):
        super().__init__(matrix, bit_depth)
        self.inv_center = (self.height - self.center[0],self.width - self.center[1])


class Image(MatrixImage):
    def __init__(self, matrix: Matrix, bit_depth: int = 8):
        super().__init__(matrix, bit_depth)

    def convolution(self, kernal: Kernal):

        # Initialize an empty output matrix
        #output_matrix = [[0 for _ in range(output_width)] for _ in range(output_height)]

        # going from 0 to the last place the kernal center can be located in so the filter is in the image
        for row in range(0,self.height - kernal.inv_center[0]):
            for col in range(0,self.width - kernal.inv_center[1]):
                # getting the values same size as the filter 
                neighborhood = [r[col:col + kernal.width] for r in self.matrix[row:row + kernal.height]]
                # print(self.matrix[row+ kernal.center[0]][col+ kernal.center[1]])

                # Perform element-wise multiplication with the kernel and sum the results
                # output_matrix[i][j] = sum(neighborhood_row[k] * kernel_row[k] for neighborhood_row, kernel_row in zip(neighborhood, kernel))
                print(neighborhood)
    

if __name__ == "__main__":
    im1 = Image.make_random(3,3)
    k = Kernal.make_random(5,5)
    print(im1)
    im1.convolution(k)


    # def convolution(self, kernal: Kernal, padding: Union[None,int] = None):

    #     # Initialize an empty output matrix
    #     #output_matrix = [[0 for _ in range(output_width)] for _ in range(output_height)]
    #     if padding == None:
    #         h = self.height
    #         w = self.width
    #         pad_matrix = self.matrix.copy()
    #     # constant or zero padding 
    #     elif isinstance(padding,int):
    #         # make bigger matrix 
    #         # center index is bigger than inverse then multiply by 2 for 2 sides
    #         h = self.height + kernal.center[0] + kernal.inv_center[0]
    #         w = self.width + kernal.center[1] + kernal.inv_center[1]
    #         pad_matrix = [[padding for _ in range(w)] for _ in range(h)]
    #         # for the new matrix i will start to copy old values staring from the kernal center
    #         for r in range(self.height):
    #             for c in range(self.width):
    #                 pad_matrix[r + kernal.center[0]][c + kernal.center[1]] = self.matrix[r][c]
    #     else:
    #         raise ValueError("padding must be None or integer")

    #     # going from 0 to the last place the kernal center can be located in so the filter is in the image
    #     for row in range(0,h - kernal.inv_center[0]):
    #         for col in range(0,w - kernal.inv_center[1]):
    #             # getting the values same size as the filter 
    #             neighborhood = [r[col:col + kernal.width] for r in pad_matrix[row:row + kernal.height]]
    #             # print(self.matrix[row+ kernal.center[0]][col+ kernal.center[1]])

    #             # Perform element-wise multiplication with the kernel and sum the results
    #             # output_matrix[i][j] = sum(neighborhood_row[k] * kernel_row[k] for neighborhood_row, kernel_row in zip(neighborhood, kernel))
    #             print(neighborhood)
