#Write a Python Program to Display the multiplication Table.

num = int(input("Enter the number for which you want to display the multiplication tables:"))

for i in range(1, 11):
    print(f" {num} X {i} = {num*i}")