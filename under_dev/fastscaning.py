import numpy as np
from DIP.helper import display_matrices, make_realistic_matrix,mat_like

# Constants
THRESHOLD1 = 25
THRESHOLD2 = 3  # Minimum number of pixels for a region to avoid merging
IMAGE_SIZE = (4, 4)  # Rows, Columns


# Fast Scanning Algorithm
def fast_scanning_algorithm(image):
    rows, cols = len(image), len(image[0])
    region_map = mat_like(image)  # Tracks region numbers
    region_means = {}  # Tracks mean intensity of each region
    current_region = 1

    # First pass: Assign pixels to regions
    for i in range(rows):
        for j in range(cols):
            pixel_value = image[i, j]

            # Check top and left neighbors
            neighbors = []
            if i > 0 and region_map[i-1, j] != 0:
                neighbors.append(region_map[i-1, j])  # Top neighbor
            if j > 0 and region_map[i, j-1] != 0:
                neighbors.append(region_map[i, j-1])  # Left neighbor

            # Find the closest matching region
            closest_region = None
            min_diff = float('inf')
            for region in neighbors:
                diff = abs(pixel_value - region_means[region])
                if diff <= THRESHOLD1 and diff < min_diff:
                    closest_region = region
                    min_diff = diff

            # Assign pixel to a region
            if closest_region is not None:
                region_map[i, j] = closest_region
                region_means[closest_region] = (
                    region_means[closest_region] * (region_map == closest_region).sum() + pixel_value
                ) / ((region_map == closest_region).sum() + 1)
            else:
                region_map[i, j] = current_region
                region_means[current_region] = pixel_value
                current_region += 1

    # # Second pass: Merge small regions
    # for region in range(1, current_region):
    #     region_size = (region_map == region).sum()
    #     if region_size < THRESHOLD2:
    #         # Find nearest region based on mean intensity
    #         nearest_region = min(
    #             range(1, current_region),
    #             key=lambda r: abs(region_means[r] - region_means[region]) if r != region else float('inf')
    #         )
    #         # Merge regions
    #         region_map[region_map == region] = nearest_region
    #         region_means[nearest_region] = (
    #             region_means[nearest_region] * (region_map == nearest_region).sum() +
    #             region_means[region] * region_size
    #         ) / ((region_map == nearest_region).sum() + region_size)

    # return region_map

# Test the algorithm
image = make_realistic_matrix(6,6)
segmented_image = fast_scanning_algorithm(image)
print("Original Image:\n", image)
print("Segmented Regions:\n", segmented_image)
