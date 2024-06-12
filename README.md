# intro

this is a small project to help me and my collegs in solving DIP (digital image processing) problems and practice it.

## getting started 
**it's better to use it in the IDLE**

start by import the package
```py
from DIP import *
```

then first you can make a random matrix using 
`make_random(width: int = 3, height: int = 3, bit_depth: int = 8) -> Matrix`

```py
width = 3
height = 3
bit_depth = 8
img1 = make_random(width, height, bit_depth)
```

then chosing a kernal from `Kernals` class
```py
kernel1 = Kernals.smoothing_filter
```

and display the matrix or many matrice using `display_matrices(list_of_matrices: List[Matrix], text: List[str] = [], coordinates: bool = False)`
```py
display_matrices( [img1, kernel1] )
```
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

[see the documentaions for more](docs.md)

# The Story

the project started as I wanted to know how the implementation of some algorithms and operations in a collage course for Digital Image Processing so it was implemented in numpy (faster and I only want to learn the concept) but after a while I said why not make it **pure python**? so no external librarys needed, and then give the code to my colleges to use it to solve any problem they want... then I said why not already make a TUI (Text based User Interface) quizzes app and adhering to the principal I tried to make it my self.

# what i learned
- how to document a python code focused
- using pydocs and focus self explanatory code
- also making docs for how to use the code
- learning a lot more about python

# to-do
-   [ ] add give me a question option (chose a thing, gives you matrix a pixel to work on or just work on the whole thing then waits for you answer and can calc time)
-   [ ] adding a steps parameter to show the steps of the algorithm
-   [ ] add bit depth for max value of segmentaion in thresholding
-   [x] resize function not like lecture

# what's next
-   implement the code using numpy and oop
-   make a text base user interface
