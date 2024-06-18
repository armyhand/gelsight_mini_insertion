import numpy as np

# Specify the path to your text file
file_path = r'object_coordinate/star_hole.txt'

# Initialize an empty list to store the (x, y, z) tuples
data_list = []

# Read the file line by line and extract (x, y, z) values
with open(file_path, 'r') as file:
    for line in file:
        # Remove parentheses and split by comma
        values = line.strip('()\n').split(',')
        # Convert values to float and append to the list
        x, y, z = map(float, values)
        data_list.append([x, y, z])

# Convert the list of lists to a Numpy array
data_array = np.array(data_list)

import matplotlib.pyplot as plt
plt.scatter(data_array[:,0], data_array[:,1])
plt.show()




