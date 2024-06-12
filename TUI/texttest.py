import curses

def main(stdscr):
    # Clear screen
    stdscr.clear()

    # Define the options for the quiz types
    quiz_types = ["Math", "Science", "History", "Literature"]

    # Initialize the selection index
    current_selection = 0

    # Main loop
    while True:
        # Clear the screen
        stdscr.clear()

        # Display the options
        for idx, quiz_type in enumerate(quiz_types):
            if idx == current_selection:
                stdscr.addstr(idx, 0, f"> {quiz_type}", curses.A_REVERSE)
            else:
                stdscr.addstr(idx, 0, f"  {quiz_type}")

        # Refresh the screen
        stdscr.refresh()

        # Get user input
        key = stdscr.getch()

        # Process the user input
        if key == curses.KEY_UP:
            current_selection = (current_selection - 1) % len(quiz_types)
        elif key == curses.KEY_DOWN:
            current_selection = (current_selection + 1) % len(quiz_types)
        elif key == ord('\n'):
            # User pressed Enter, break the loop
            break

    # Return the selected quiz type
    return quiz_types[current_selection]

if __name__ == "__main__":
    selected_quiz_type = curses.wrapper(main)
    print(f"You selected: {selected_quiz_type}")
