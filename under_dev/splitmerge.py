from DIP.helper import Matrix
from typing import List

def split_and_merge_regions(matrix: Matrix, region_size: int=25) -> List[Matrix]:
    """
    Splits an image into regions and merges them based on similarity.

    Parameters:
        `matrix` (Matrix): The input matrix representing an image.
        `region_size` (int, optional): The size of each region. Defaults to 25.

    Returns:
        List[Matrix]: A list of region matrices merged based on similarity.
    """
    regions = []
    height, width = len(matrix), len(matrix[0])

    # Split the image into regions
    for i in range(0, height, region_size):
        for j in range(0, width, region_size):
            region = [row[j:j+region_size] for row in matrix[i:i+region_size]]
            regions.append(region)

    merged_regions = []
    while len(regions) > 1:
        # Find the most similar regions
        min_distance = float('inf')
        pair = (None, None)
        
        for i in range(len(regions)):
            for j in range(i+1, len(regions)):
                distance = sum(sum(abs(a - b) for a, b in zip(r1, r2)) for r1, r2 in zip(regions[i], regions[j]))
                
                if distance < min_distance:
                    min_distance = distance
                    pair = (i, j)
        
        # Merge the most similar regions
        merged_region = [[sum(a + b) // 2 for a, b in zip(r1, r2)] for r1, r2 in zip(regions[pair[0]], regions[pair[1]])]
        merged_regions.append(merged_region)

        # Remove the merged regions from the list
        del regions[pair[0]]
        regions.pop(pair[1])

    return merged_regions
