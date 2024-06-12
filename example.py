from DIP import *

# for this you need to install matplotlib
# pip install matplotlib

def plot_morphology(images, titles):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for ax, img, title in zip(axes, images, titles):
        # setting vmin=0 and vmax=1, you ensure that 0 is mapped to white and 1 is mapped to black,
        # as intended when using the gray_r colormap.
        ax.imshow(img, cmap='gray_r', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')

        # Add border around each element
        num_rows, num_cols = len(img),len(img[0])
        for i in range(num_rows):
            for j in range(num_cols):
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', linewidth=1)
                ax.add_patch(rect)

    plt.show()

# --------------------------------------------------------------------------------

bit_dipth = 4

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

# im = [[0,0,0,0,0,0,0,0],
#       [0,0,0,1,1,0,0,0],
#       [0,0,1,1,1,1,0,0],
#       [0,1,1,1,1,1,1,0],
#       [0,1,1,1,1,1,1,0],
#       [0,0,1,1,1,1,0,0],
#       [0,0,0,1,1,0,0,0],
#       [0,0,0,0,0,0,0,0]]

# # im = [[0,0,0,0],
# #       [0,1,0,0],
# #       [0,0,0,0]]

# se1 = [[0,0,0],
#       [0,1,0],
#       [0,0,0]]

# se2 = [[0,0,0],
#       [0,1,0],
#       [0,0,0]]

# # se = complement(se,1)
# result = hit_or_miss(im,se1,se2)

# plot_morphology([im,se1,se2,result],['image','se1','se2','hole filling'])

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

im = [[1,1,1,1,0,0],
      [1,0,0,1,1,0],
      [0,1,0,0,1,0],
      [0,0,1,1,1,0],
      [0,0,0,0,0,0]]

se1 = [[1,0,1],
      [1,1,0],
      [0,0,1]]


# se = complement(se,1)
# x1 = hole_filling(im,se1,(2,3),1)
# x2 = hole_filling(im,se1,(2,3),2)
# x3 = hole_filling(im,se1,(2,3),3)
xs = [*hole_filling_X_gen(im,se1,(2,3),3)]

plot_morphology(xs,['X 0','X 1','X 2','X 3'])
