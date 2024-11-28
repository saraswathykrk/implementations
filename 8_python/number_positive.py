#Write a Python Program to Check if a Number is Positive, Negative or Zero.

num = int(input("Enter a number:"))

if num > 0:
    print(f"Entered number {num} is a positive num.")
elif num < 0:
    print(f"Entered number {num} is a negative num.")
else:
    print(f"Entered number is {num}.")