#Write a Python program to swap two variables.

a = input("Enter the value of the first variable:")
b = input("Enter the value of the second variable:")
print(f"Original values: a : {a} , b : {b}")

temp = a
a = b
b = temp
print(f"Swapped values: a : {a} , b : {b}")