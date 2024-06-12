from DIP import *
from DIP.DIP_NH_pure import _neighborhoods_of_pixel, _neighborhood_with_center


b = 4

# t = [1,0,5,0,6,6,2,5]
# equalized_matrix = histogram_matching(matrix,target_histogram=t, show=True)
# display_matrices([equalized_matrix],text=["Histogram Matching Result:"])
# im1 = make_random(4,5,b)


# im1 = [[0,4,2,6,4],
#        [4,2,4,5,6],
#        [2,4,0,4,6],
#        [6,6,6,0,6]]
# im2 = _neighborhoods_of_pixel(im1,(0,1),(3,2))
# display_matrices([im1,im2],coordinates=True)

# matrix = [[1, 2],
#            [3, 4]]
# kernel_dim = (2, 2)
# gen = _neighborhood_with_center(matrix, kernel_dim)
# for neighborhood, center in gen:
#        print("Neighborhood:", neighborhood)
#        print("Center coordinates:", center)

# matrix = [[1, 2, 3],
#           [4, 5, 6],
#           [7, 8, 9]]
# kernel = [[1, 0, -1],
#           [1, 0, -1],
#           [1, 0, -1]]
# result = convolute(matrix, kernel,pad_with=0)
# print(result)

# matrix = [[1, 2, 3],
#           [4, 5, 6],
#           [7, 8, 9]]
# kernel_dim = (3, 3)
# rank = 3
# result = rank_order_filter(matrix, kernel_dim, rank, pad_with = 0)

# print(result)

# matrix = [[105, 102, 100],
#           [107, 120, 103],
#           [110, 108, 104]]

# kernel_dim = (2, 2)

# result = adaptive_filter(matrix, kernel_dim,pad_with=0)

# display_matrices([matrix, result])


# matrix = [[0,4,2,6,4],
#           [4,2,4,5,6],
#           [2,4,0,4,6],
#           [6,6,6,0,6]]

# target_histogram = [1,0,5,0,6,6,2,5]

# matched_matrix = histogram_matching(matrix, target_histogram, bit_depth=3, show=True)

# display_matrices([matched_matrix],text=["Histogram Matching Result:"])

im = [[1,0,0,0,0],
      [0,0,0,0,0],
      [0,0,1,1,0],
      [0,1,1,1,0],
      [1,1,1,1,0],
      [1,1,0,1,0]]

se = [[0,1,0],
      [1,0,1],
      [0,1,0]]

result = erosion(im,se)

plot_morphology([im,se,result],['image','se','result'])

# display_matrices([im, result])

# mat = [[180,245,250,220],[210,225,215,215],[218,230,220,212],[222,215,218,210]]
# result = automatic_threshold(mat,.9,show_steps=True)

# mat = [[50,60,110,20,200],
#         [40,50,30,50,90],
#         [35,70,150,50,120],
#         [40,150,200,160,50],
#         [30,200,120,120,40]]

# result = convolute(mat,Kernals.laplacian_operator,pad_with=0)

# display_matrices([mat,result],['original','result'], coordinates=True)