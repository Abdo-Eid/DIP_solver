import os
import DIP as d

def clear_terminal():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def select_problem_type():
    problem_types = ["Math", "Science", "History", "Literature"]

    while True:
        print("Select the type of problem:")
        for index, problem in enumerate(problem_types, start=1):
            print(f"{index}. {problem}")

        selected_index = input("Choose a problem type number (type 'quit' to exit): ")

        clear_terminal()

        if selected_index.lower() == 'quit':
            print("Goodbye!")
            break

        try:
            selected_index = int(selected_index)
            if 1 <= selected_index <= len(problem_types):
                selected_problem = problem_types[selected_index - 1]
                if selected_problem == 'Math':
                    d.display_matrices([d.make_random()])
                else:
                    print(f"You selected: {selected_problem}")
            else:
                print("Invalid choice. Please select a valid problem type number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    select_problem_type()
