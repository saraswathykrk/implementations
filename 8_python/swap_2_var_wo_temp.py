# Write a Python program to swap two variables without temp variable.

a = int(input("Enter the first number:"))
b = int(input("Enter the second number:"))

print(f"Original numbers: a = {a}, b = {b}")

#Swapping without temp variable:
a, b = b, a

print(f"After swapping: a = {a}, b = {b}")