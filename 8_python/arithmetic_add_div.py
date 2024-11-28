#Write a Python program to do arithmetical operations addition and division.
#Addition

num1 = float(input("Enter the first number for addition:"))
num2 = float(input("Enter the second number for addition:"))
sum = num1 + num2
print(f" sum: {num1} + {num2} = {sum}")


#Division

num3 = float(input("Enter the dividend for division:"))
num4 = float(input("Enter the divisor for division:"))

if num4 == 0:
    print("Error: Division by zero is not allowed.")
else:
    div = num3 / num4
    print(f" Division: {num3} / {num4} = {div} ")