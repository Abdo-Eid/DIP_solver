import random
from typing import List, Tuple
import matplotlib.pyplot as plt



def get_shape_path(grid_rows, grid_cols):
    points = set()  # To store points
    start_point = None  # To store the start point
    
    def on_click(event):
        """Handle mouse click events to add/remove points or set the start point."""
        nonlocal start_point

        # Check if click is within the grid
        if event.xdata is None or event.ydata is None:
            return

        # Snap to nearest grid intersection
        x = round(event.xdata)
        y = round(event.ydata)

        # Ensure click is within bounds (note the swap of x and y)
        if not (0 <= y <= grid_rows and 0 <= x <= grid_cols):
            return

        clicked_point = (y, x)  # Store as (y, x)

        if event.button == 1:  # Left-click to add/remove points
            if clicked_point in points:  # Remove the point if it exists
                points.remove(clicked_point)
                if start_point == clicked_point:  # Reset start point if removed
                    start_point = None
            else:  # Add the point
                points.add(clicked_point)
            update_plot()
        elif event.button == 3:  # Right-click to set as start point
            if clicked_point in points:
                start_point = clicked_point
                update_plot()
            else:
                print("You must first create a point before setting it as the start point.")


    def on_key(event):
        """Handle keyboard events to trigger chain code calculation."""
        if event.key == "p":
            print("Points:", sorted(points))
            print(f"Start point: {start_point}")


    def update_plot():
        """Update the plot based on the current points and start point."""
        plt.clf()

        # Plot the grid
        for i in range(grid_rows + 1):
            plt.plot([0, grid_cols], [i, i], color="black", linewidth=0.5)  # Horizontal lines
        for j in range(grid_cols + 1):
            plt.plot([j, j], [0, grid_rows], color="black", linewidth=0.5)  # Vertical lines

        # Plot points
        for point in points:
            plt.plot(point[1], point[0], "ko")  # Plot as (x, y), but store as (y, x)

        # Plot start point
        if start_point:
            plt.plot(start_point[1], start_point[0], "bo", markersize=15, markerfacecolor="none")  # Circle around the start point

        # Set plot limits and labels
        plt.xlim(-1, grid_cols + 1)
        plt.ylim(grid_rows + 1, -1)  # Inverted y-axis
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xticks(range(grid_cols + 1))
        plt.yticks(range(grid_rows + 1))
        plt.xlabel("y")
        plt.ylabel("x")
        plt.grid(False)  # Disable auto grid

        plt.draw()

    # Setup plot
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    update_plot()
    plt.show()
    return points, start_point


def get_chain_code_4_directions(points: List[Tuple[int, int]], start_point: Tuple[int, int] = None) -> str:
    if not points:
        return ""

    # Define directions according to your convention (clockwise)
    # Right (0), Down (3), Left (2), Up (1)
    directions = [
        (0, 1),   # 0: Right (0°)
        (-1, 0),  # 1: Up (90°)
        (0, -1),  # 2: Left (180°)
        (1, 0)    # 3: Down (270°)
    ]
    
    starting_point = min(points, key=lambda p: (p[0], p[1]))
    start_point = start_point or starting_point
    
    points_set = set(points)  # Set for fast lookups
    if start_point not in points_set:
        raise ValueError("start point must be on the boundary of the shape")
    visited = []           # Keep track of visited points
    chain_code = ""
    current = starting_point
    
    while True:
        found_point = False
        
        for i, (dy, dx) in enumerate(directions):
            next_point = (current[0] + dy, current[1] + dx)
            
            while next_point in points_set and next_point not in visited:
                chain_code += str(i)      # Add direction to chain code
                visited.append(next_point)   # Mark as visited
                current = next_point      # Move to next point
                found_point = True
                
                next_point = (current[0] + dy, current[1] + dx)  # Continue in the same direction
            
            if found_point:
                break
        
        if not found_point:
            break
        
        if current == starting_point:
            visited.insert(0, starting_point)
            break
    start_idx = visited.index(start_point)
    n = len(chain_code)
    dc = chain_code*2

    return dc[start_idx:start_idx+n]

# get_shape_path(5,5)

##################################
def generate_random_chain_code(length: int, num_directions: int = 4) -> str:
    # Define possible symbols for the chain code
    if num_directions == 8:
        symbols = "01234567"
    if num_directions == 4:
        symbols = "0123"
    # Generate a random sequence of the specified length
    return ''.join(random.choice(symbols) for _ in range(length))

def normalize_chain_code(chaincode:str) -> str:
    n = len(chaincode)
    doubled = chaincode*2
    smallest_value = int(chaincode)
    smallest_indx = 0
    for i in range(n):
        current = int(doubled[i:i+n])
        if current < smallest_value:
            smallest_value = current
            smallest_indx = i
    return doubled[smallest_indx:smallest_indx+n]

def diff_chain_code(chaincode:str, directions: int = 4) -> str:
    out = ""
    for i,_ in enumerate(chaincode):
        out += str((int(chaincode[i]) - int(chaincode[i-1])) % directions)
    return out

def shaped_num(chaincode:str, directions: int = 4) -> str:
    return normalize_chain_code(diff_chain_code(chaincode, directions))



# Generate a random chain code with 8 symbols
directions = 4


random_chain_code = generate_random_chain_code(7,directions)
print("Random Chain Code:", random_chain_code)

input("Press Enter to continue...")

normalized_random_chain_code = normalize_chain_code(random_chain_code)
print("Normalized Random Chain Code:", normalized_random_chain_code)

print("diffed chain code: ", diff_chain_code(random_chain_code, directions))

shaped_chain = shaped_num(random_chain_code, directions)
print("Shaped Chain Code:", shaped_chain)

