# implemented algorithms and the corsponding function
basic image operaions (*, +, /, -)
image logical operations (And, OR, XOR)
more image operations
Solarization (dark part or bright part) 
Gamma correction 
Histogram stretching 
Histogram equalization 
Histogram matching 
Smoothing filter 
Weight smoothing filter 
High pass filter 
An outlier method 
Adaptive filter 
Automatic threshold 
Robert operator 
Prewitt operator 
Sobel operator 
Laplacian operator 
Dilation 
Erosion 
Opening 
Closing 
Boundary Extraction (internel / external / morphological gradient) 
Hole filling 
Hit or miss

# getting started 
first you can make a random matrix using `make_random` function
```py
make_random(width: int = 3, height: int = 3, bit_depth: int = 8) -> Matrix
```

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

and display the matrix or many matrice using `display_matrices` function
```py
display_matrices(list_of_matrices: List[Matrix], text: List[str] = [], coordinates: bool = False)
```
```py
display_matrices([img1,kernel1], coordinates = False)
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