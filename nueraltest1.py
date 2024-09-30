import matplotlib.pyplot as plt
import numpy as np


weight_1_1 = 1.0
weight_2_1 = 1.0
weight_1_2 = 1.0
weight_2_2 = 1.0
#weights the values to output 1 and then output 2
input1=None
input2=None
def Classification(input1,input2):
    output1=(input1*weight_1_1) + (input2*weight_2_1)
    output2=(input2*weight_1_2) + (input2*weight_2_2)

    if (output1>output2):
        return 0
    else:
        return 1
    
input1_values = np.linspace(1, 1, 10)
input2_values = np.linspace(1, 1, 10)

# Create a meshgrid to evaluate the function on a grid
input1_grid, input2_grid = np.meshgrid(input1_values, input2_values)

# Classify each point in the grid
classification_results = np.array([Classification(x, y) for x, y in zip(np.ravel(input1_grid), np.ravel(input2_grid))])
classification_results = classification_results.reshape(input1_grid.shape)
plt.figure(figsize=(8, 6))
plt.contourf(input1_grid, input2_grid, classification_results, cmap='coolwarm', alpha=0.8)

# Adding labels and title
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('Classification Results')
plt.colorbar(label='Class')

# Show the plot
plt.show()