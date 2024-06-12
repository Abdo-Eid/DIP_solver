from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

# thype of the problem

# 



def main():
    question_types = ['Addition', 'Subtraction', 'Multiplication', 'Division']
    question_completer = WordCompleter(question_types, ignore_case=True)

    while True:
        question_type = prompt('Select a question type: ', completer=question_completer)
        
        if question_type not in question_types:
            print("Invalid question type. Please choose from the list.")
            continue

        # Generate a random 3x3 matrix as an exampl

        # Wait for user's answer (you can implement specific logic based on question type)
        user_answer = prompt('\nEnter your answer: ')
        
        # Here you would implement the logic to check the user's answer
        print(f"You answered: {user_answer}\n")
        
        # Option to quit or continue
        if prompt('Do you want to continue? (yes/no): ').lower() != 'yes':
            break

if __name__ == "__main__":
    main()
