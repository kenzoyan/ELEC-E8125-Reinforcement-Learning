import numpy as np
from matplotlib import pyplot as plt
from numpy import random
import seaborn as sb


def plot_value_function(value_function):
    sb.heatmap(np.mean(value_function, axis=(1, 3)))
    plt.show()
    
values=np.load("value_func.npy")
q_grid=np.load("q_values.npy")

print("q_grid",np.sum(q_grid))
# Save the Q-value array


# Calculate the value function
values = np.amax(q_grid,axis=4) # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
np.save("value_func.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY


# Plot the heatmap
# TODO: Plot the heatmap here using Seaborn or Matplotlib
plot_value_function(values)