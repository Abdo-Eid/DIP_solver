# for this you need to install matplotlib
# pip install matplotlib
from .helper import Matrix,List
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

def animate_convex_hull(convex_hull_gen, interval=1000, user_control=False, show_prev_in_gray=True):
    """
    Animate the convex hull generation process, optionally displaying the previous step in gray.

    Parameters:
        convex_hull_gen: A generator function that yields tuples of the form 
                         (structure_num, iteration_num, matrix). The `matrix` should be a 2D list
        interval (int): Delay between frames in milliseconds for automatic mode. Defaults to 1000 ms.
        user_control (bool): If True, allows manual stepping through frames using key presses instead 
                             of automatic playback. Press the right arrow key to advance to the next frame.
        show_prev_in_gray (bool): If True, displays the previous step in gray behind the current step 
                                  for better visual comparison. Defaults to False.

    Example:
        >>> gen = convex_hull_gen(input_matrix, structures)
        >>> animate_convex_hull(gen, interval=500, user_control=True)
    """
    fig, ax = plt.subplots()
    
    # Initialize with the first matrix from the generator
    structure_num, iteration_num, matrix = next(convex_hull_gen)
    prev_matrix = None  # Will hold the previous matrix for displaying in gray

    # Display the current matrix in color and add text for metadata
    im_color = ax.imshow(matrix, cmap='gray_r')  # Use a color map for the current step
    
    # Optional: Set up gray image for the previous step
    im_gray = ax.imshow(matrix, cmap='gray_r', alpha=0.5) if show_prev_in_gray else None
    if im_gray:
        im_gray.set_visible(False)  # Hide initially since there's no previous frame
    
    text = ax.text(0.02, 0.95, f'Structure: {structure_num}, Iteration: {iteration_num}', 
                   transform=ax.transAxes, color="red", fontsize=12)

    # Update function to switch frames
    def update(_):
        nonlocal prev_matrix  # Access the previous matrix

        try:
            # Get the next frame data
            structure_num, iteration_num, matrix = next(convex_hull_gen)
            
            # Update previous step in gray if enabled and exists
            if show_prev_in_gray and prev_matrix is not None:
                im_gray.set_array(prev_matrix)
                if iteration_num != 0:
                    im_gray.set_visible(True)
                else:
                    im_gray.set_visible(False)
            prev_matrix = matrix  # Update the previous matrix to the current one

            # Update the color image and text for the current step
            im_color.set_array(matrix)
            text.set_text(f'Structure: {structure_num}, Iteration: {iteration_num}')
        except StopIteration:
            pass
        return [im_gray, im_color, text] if show_prev_in_gray else [im_color, text]

    # Manual stepping function with key press, if user_control is True
    def on_key(event):
        if event.key == 'right':
            update(None)
            plt.draw()

    if user_control:
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()
    else:
        ani = animation.FuncAnimation(fig, update, frames=None, cache_frame_data=False, interval=interval, blit=False, repeat=False)
        plt.show()
