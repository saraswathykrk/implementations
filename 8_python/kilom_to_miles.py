# Write a Python program to convert kilometers to miles.
 
km = float(input("Enter the kilometres you want to convert:"))

# Conversion factor: 1 kilometer = 0.621371 miles
conversion_factor = 0.6

miles = km * conversion_factor

print(f"{km} kilometres is equal to {miles} miles.")