# Calculator Program
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if a == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Main Program
print("Welcome to the Calculator!\nLet's start typing numbers and operations.")

# Display the four basic operations menu
operations = {
    'add': (add, 'Add'),
    'subtract': (subtract, 'Subtract'),
    'multiply': (multiply, 'Multiply'),
    'divide': (divide, 'Divide')
}

# Get user input for operation
selected_operation = input("Select an operation: ").strip()
if selected_operation not in operations:
    print("Invalid operation. Please try again.")
else:
    op, prompt = operations[selected_operation]

    # Display the first number
    num1 = int(input(f"Enter the first number: "))
    print(f"The first number you entered is {num1}.")

    # Get the second number based on the selected operation
    if selected_operation == 'add':
        num2 = int(input("Enter the second number for addition: "))
    elif selected_operation == 'subtract':
        num2 = int(input("Enter the second number for subtraction: "))
    elif selected_operation == 'multiply':
        num2 = int(input("Enter the second number for multiplication: "))
    else:
        num2 = int(input("Enter the second number for division: "))

    # Perform the operation and display result
    if op == 'add':
        print(f"{num1} + {num2} = {op(num1, num2)}")
    elif op == 'subtract':
        print(f"{num1} - {num2} = {op(num1, num2)}")
    elif op == 'multiply':
        print(f"{num1} * {num2} = {op(num1, num2)}")
    else:  # divide
        result = op(num1, num2)
        if result == 0:
            print("Cannot divide by zero.")
        else:
            print(f"{num1} / {num2} = {result}")

# Close the main menu after all operations are done
print("\nThank you for using the calculator!")