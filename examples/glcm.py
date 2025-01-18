from DIP.cv import calculate_glcm, agg_glcm
from DIP.helper import display_matrices, make_random
test_matrix = make_random(4,4, 2)

display_matrices([test_matrix])

input("any to skip")

distances = [1]
angles = [135]

for distance in distances:
    for angle in angles:

        int_glcm = calculate_glcm(test_matrix, distance=distance, theta=angle)
        
        norm_glcm = calculate_glcm(test_matrix, distance=distance, theta=angle, normalize=True)

        display_matrices([int_glcm, norm_glcm], [f"GLCM (distance {distance}, angle {angle} degrees):", "Normalized GLCM"])

Contrast, Homogeneity, Entropy = agg_glcm(norm_glcm)

print("Contrast: ", Contrast)
print("Homogeneity: ", Homogeneity)
print("Entropy: ", Entropy)