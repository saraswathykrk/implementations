# Write a Python Program to Make a Simple Calculator with 4 basic mathematical operations.

#Addition
def add(a, b):
    return (a+b)

#Subtraction
def sub(a, b):
    return (a-b)

#Multiplication
def mul(a, b):
    return (a*b)

#Division
def div(a, b):
    return (a/b)

print("Select Operation:")
print("1. Add")
print("2. Subtract")
print("3. Multiply")
print("4. Divide")

while True:
    choice = int(input("Enter choice(1, 2, 3, 4): "))

    if choice in (1, 2, 3, 4):
        try:
            num1 = float(input("Enter the first number: "))
            num2 = float(input("Enter the second number: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if choice == 1:
            print(num1, "+", num2, "=", add(num1, num2))
        elif choice == 2:
            print(num1, "-", num2, "=", sub(num1, num2))
        elif choice == 3:
            print(num1, "", num2, "=", mul(num1, num2))
        elif choice == 4:
            print(num1, "+", num2, "=", div(num1, num2))

        

            

