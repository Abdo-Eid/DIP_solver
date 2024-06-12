# for this you need to install matplotlib
# pip install matplotlib
from .helper import Matrix,List
import matplotlib.pyplot as plt

def plot_morphology(images: List[Matrix], titles: List[str] = None, draw_border=True, show_numbers=False, show_axis=False, figure_scale=0.7):
    """
    Plot morphological images with optional features and size scaling.
    
    Parameters:
        `images` (list of matrices): List of images to display.
        `titles` (list of str, optional): Titles for each image. If None, no titles are shown.
        `draw_border` (bool): If True, draws a border around each pixel.
        `show_numbers` (bool): If True, shows the value of each pixel.
        `show_axis` (bool): If True, shows numbering on x and y axes.
        `scale` (float): Scale factor for figure size (0 to 1, default=0.7).
    """
    # Validate scale parameter
    if not 0 < figure_scale <= 1:
        raise ValueError("Scale parameter must be between 0 and 1")
    
    # Calculate width ratios for consistent scaling
    width_ratios = [len(img[0]) for img in images]
    total_width = 4 * (sum(width_ratios) / max(width_ratios))
    
    # Apply scale to figure size
    scaled_width = total_width * figure_scale
    scaled_height = 4 * figure_scale

    # Create figure with proper width ratios and scaled size
    fig, axes = plt.subplots(1, len(images), figsize=(scaled_width, scaled_height),
                            gridspec_kw={'width_ratios': width_ratios})
    
    if len(images) == 1:
        axes = [axes]

    if titles is None:
        titles = [i for i in range(1, len(images) + 1)]

    for ax, img, title in zip(axes, images, titles):
        # Display image with consistent scaling
        ax.imshow(img, cmap='gray_r', vmin=0, vmax=1)
        if title:  # Only set title if it's not an empty string
            ax.set_title(title)
        
        if show_axis:
            # Add axis numbering
            ax.set_xticks(range(len(img[0])))
            ax.set_yticks(range(len(img)))
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            
        if show_numbers:
            # Add pixel values
            for i in range(len(img)):
                for j in range(len(img[0])):
                    text_color = 'white' if img[i][j] == 1 else 'black'
                    # Scale font size with figure size
                    fontsize = 10 * figure_scale
                    ax.text(j, i, str(img[i][j]), ha='center', va='center', 
                           color=text_color, fontsize=fontsize)

        if draw_border:
            num_rows, num_cols = len(img), len(img[0])
            for i in range(num_rows):
                for j in range(num_cols):
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, 
                                      edgecolor='black', linewidth=max(0.5, 1 * figure_scale))
                    ax.add_patch(rect)
    
    plt.tight_layout()
    plt.show()
