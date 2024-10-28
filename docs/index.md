# Digital Image Processing Helper

## Introduction

This project provides tools for solving Digital Image Processing (DIP) problems, aimed at helping me and my colleagues practice and understand core concepts in DIP.

## Installation

1. **Clone the Repository**

    ``` sh
    git clone <repository_url>
    ```

2. **Navigate to the Project Directory**

    ``` sh
    cd DIP_solver
    ```

3. **Install the Package**  
   Run the following command to install the package and dependencies in your environment:

    ``` sh
    pip install .
    ```

## Getting Started

> **Note:** Using this project in IDLE or an interactive Python environment is recommended.

1. **Import the Package**  
   Start by importing the package:

    ```python
    from DIP.helper import make_random
    ```

2. **Create a Random Matrix**  
   You can generate a random matrix with the function `make_random(width: int = 3, height: int = 3, bit_depth: int = 8) -> Matrix`.

    ```python
    width = 3
    height = 3
    bit_depth = 8
    img1 = make_random(width, height, bit_depth)
    ```

3. **Choose a Kernel**  
   Import a predefined kernel from the `Kernels` class:

    ```python
    from DIP.neighborhood_operations import Kernals
    kernel1 = Kernals.smoothing_filter
    ```

4. **Display Matrices**  
   Display one or more matrices using `display_matrices(list_of_matrices: List[Matrix], text: Optional[List[str]] = None, coordinates: bool = False)`:

    ```python
    display_matrices([img1, kernel1])
    ```

    Sample output:

    ```
    Matrix 1:
    [ 125 208 136 ]
    [ 130   1 240 ]
    [  93 207 170 ]

    Matrix 2:
    [ 0.111 0.111 0.111 ]
    [ 0.111 0.111 0.111 ]
    [ 0.111 0.111 0.111 ]
    ```

## The Story

The project started because I wanted to understand the implementation of some algorithms and operations for a college course on Digital Image Processing. At first, I built it in NumPy—faster, and I just wanted to learn the concepts. But then I thought, why not make it **pure Python**? This way, there are no external libraries needed, and I could share the code with my classmates so they could use it to solve any problem they wanted.

Then the idea grew: why not make a TUI (Text-based User Interface) and quiz app that sticks to this DIY principle? I decided to build it entirely myself to deepen my understanding and give my classmates a useful, accessible tool.

## What I Learned

-   Writing clear and helpful documentation
-   Using Pydoc and focusing on self-explanatory code
-   Creating guides on using code effectively
-   Gaining deeper knowledge of Python

## What's Next

-   Reimplement the project using NumPy and OOP principles
-   make a text base user interface (TUI)
