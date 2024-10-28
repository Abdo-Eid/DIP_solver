from DIP import *
from DIP import plugins
from DIP.neighborhood_operations import pad
# --------------------------------------------------------------------------------

bit_dipth = 4



# B1 = [
#     [1,'x','x'],
#     [1, 0, 'x'],
#     [1,'x','x']
# ]

# B2 = helper.rotate_90(B1)
# B3 = helper.rotate_90(B2)
# B4 = helper.rotate_90(B3)

# im = [
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ]
# c = morphology.convex_hull_gen(im,[B1,B2,B3,B4])
# plugins.animate_convex_hull(c,show_prev_in_gray=True,interval=500)


# im = helper.make_realistic_matrix()

# helper.display_matrices([im])


# im = [[0,0,0,0,0,0,0,0],
#       [0,0,0,1,1,0,0,0],
#       [0,0,1,1,1,1,0,0],
#       [0,1,1,1,1,1,1,0],
#       [0,1,1,1,1,1,1,0],
#       [0,0,1,1,1,1,0,0],
#       [0,0,0,1,1,0,0,0],
#       [0,0,0,0,0,0,0,0]]

# se1 = [[0,1,0],
#       [1,1,0],
#       [0,1,0]]

# se2 = [[1,0,0],
#       [0,0,0],
#       [1,0,0]]

# result = hit_or_miss(im, se1, se2)

# plot_morphology([im, se1, se2, result], ['Image', 'SE1', 'SE2', 'Hit or Miss'],
#                 draw_border=True, show_numbers=True, show_axis=True, figure_scale=0.5)
# display_matrices([im, result])

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

# mat = [[180,245,250,220],[210,225,215,215],[218,230,220,212],[222,215,218,210]]
# result = automatic_threshold(mat,.9,show_steps=True)

# mat = [[50,60,110,20,200],
#         [40,50,30,50,90],
#         [35,70,150,50,120],
#         [40,150,200,160,50],
#         [30,200,120,120,40]]

# result = convolute(mat,Kernals.laplacian_operator,pad_with=0)

# display_matrices([mat,result],['original','result'], coordinates=True)

# mat=[[30,10,15,15,15],
#     [30,20,15,15,20],
#     [20,10,20,10,20],
#     [25,25,24,10,20],
#     [30,20,20,20,20]]

# # se2 = [[0,0,0],
# #       [0,1,0],
# #       [0,0,0]]

# se2 = [[0,0],
#       [0,1]]

# # result = convolute(mat,Kernals.robert_operator,None,5)
# result = robert_operator(mat,5,0)
# display_matrices([mat,result],['original','result'], coordinates=True)

# im = [[1,1,1,1,0],
#       [1,0,0,1,1],
#       [0,1,0,0,1],
#       [0,0,1,1,1],
#       [0,0,0,0,0]]

# se1 = [[1,0,1],
#       [1,1,0],
#       [0,0,1]]

# xs = [*hole_filling_X_gen(im,se1,(2,3),3)]

# plot_morphology(xs)


# im = [
#         [3, 0],
#         [2, 1],
#         [5, 3],
#         [4, 1]
#     ]

# se = [
#         [1],
#         [1],
#         [1],
#         [1]
#     ]

# display_matrices([rank_order_filter(im,(2,3),1,pad_with=0)])
