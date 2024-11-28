#Write a Python program to convert Celsius to Fahrenheit.

cel = float(input("Enter the temperature in Celsius:"))

# Conversion formula: Fahrenheit = (Celsius * 9/5) + 32

fah = (cel * 9 / 5) + 32

print(f"{cel} degrees Celsius is equal to {fah} degress Fahrenheit")