# for this you need to install matplotlib
# pip install matplotlib
from typing import Tuple
from .helper import Matrix,List
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def _convert_to_numeric(img):
    """
    Convert image with 'x' values to numeric format.
    Private helper function.
    """
    return [[0 if val == 'x' else float(val) for val in row] for row in img]

def plot_morphology(images: List[Matrix], titles: List[str] = None, draw_border=True, show_numbers=False, show_axis=False, figure_scale=0.7):
    """
    Plot morphological images with optional features and size scaling.
    Handles structuring elements with 'x' values by plotting them as text overlays.
    
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
        titles = [str(i) for i in range(1, len(images) + 1)]

    for ax, img, title in zip(axes, images, titles):
        # Convert image to numeric format for display
        numeric_img = _convert_to_numeric(img)
        
        # Display image with consistent scaling
        ax.imshow(numeric_img, cmap='gray_r', vmin=0, vmax=1)
        if title:  # Only set title if it's not an empty string
            ax.set_title(title)
        
        if show_axis:
            # Add axis numbering
            ax.set_xticks(range(len(img[0])))
            ax.set_yticks(range(len(img)))
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            
        # Add pixel values and 'x' markers
        for i in range(len(img)):
            for j in range(len(img[0])):
                val = img[i][j]
                if val == 'x':
                    # Plot 'x' in red
                    ax.text(j, i, 'x', ha='center', va='center', 
                           color='black', fontsize=50 * figure_scale, fontweight='bold')
                elif show_numbers:
                    # Plot numeric values
                    text_color = 'white' if val == 1 else 'black'
                    ax.text(j, i, str(val), ha='center', va='center', 
                           color=text_color, fontsize=10 * figure_scale)

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
                                  for better visual comparison. Defaults to True.

    Example:
        >>> gen = convex_hull_gen(input_matrix, structures)
        >>> animate_convex_hull(gen, interval=500, user_control=True)
    """

    fig, ax = plt.subplots()
    frames_list: List[Tuple[int, int, Matrix]] = list(convex_hull_gen)
    frame_num = len(frames_list)
    
    if frame_num == 0:
        raise ValueError("No frames provided by the generator")

    # Initialize with the first matrix from the generator
    structure_num, iteration_num, matrix = frames_list[0]
    current_frame = 0  # Track the current frame index
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
        nonlocal current_frame, prev_matrix
        # Special last frame to show first matrix in color and last matrix in gray
        if show_prev_in_gray and current_frame == frame_num:
            im_color.set_array(frames_list[0][2])
            if im_gray:
                im_gray.set_array(frames_list[-1][2])  # Show last matrix in gray
                im_gray.set_visible(True)
            text.set_text("Final")
            return [im_gray, im_color, text] if show_prev_in_gray else [im_color, text]
        
        if current_frame < frame_num:
            structure_num, iteration_num, matrix = frames_list[current_frame]
                
            # Update previous step in gray if enabled and exists
            if show_prev_in_gray and prev_matrix is not None:
                im_gray.set_array(prev_matrix)
                if iteration_num != 0:  # Don't show gray image at start of new structure
                    im_gray.set_visible(True)
                else:
                    im_gray.set_visible(False)

            # Update the color image and text for the current step
            im_color.set_array(matrix)
            text.set_text(f'Structure: {structure_num}, Iteration: {iteration_num}')
            
            prev_matrix = matrix  # Store current matrix for next frame
            current_frame += 1  # Increment frame counter

            return [im_gray, im_color, text] if show_prev_in_gray else [im_color, text]

    # Manual stepping function with key press
    def on_key(event):
        # stop_condition = current_frame <= frame_num if show_prev_in_gray else current_frame < frame_num
        if event.key == 'right':
            update(None)
            plt.draw()

    if user_control:
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()
    else:
        ani = animation.FuncAnimation(
            fig, 
            update, 
            frames=frame_num, 
            interval=interval,
            blit=False,
            repeat=False
        )
        plt.show()
        # to save you must comment plt.show() to save all frames
        # ani.save("animation.mp4", writer="ffmpeg", fps=1)