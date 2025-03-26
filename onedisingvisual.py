import numpy as np
import random
import matplotlib.pyplot as plt

#function creates random intitial lattice of some size given as parameter

def rand_array(size):

    array = np.zeros(size)
    for i in range(array.size):
        rand_num = random.random()
        if rand_num < 0.5:
            array[i] = -1
        else:
            array[i] = 1

    return array

#function to calculate the energy change of flipping a given dipole

def deltaU(array, i):

    if i == 0:
        left = array[array.size - 1]
        right = array[1]
    if i == array.size - 1:
        right = array[0]
        left = array[array.size - 2]
    else:
        left = array[i-1]
        right = array[i+1]
    energy_diff = 2*array[i]*(left+right)

    return energy_diff


#this creates a random initial state and sets some chosen temperature 

size = 200
temp = 0.01
state = rand_array(size)
state_initial = np.copy(state)

print(state_initial)

#main loop that interates a flips diploles one at a time 

for iter in range(100*size):

    #select random dipole 
    i = int(random.random()*(size))

    #calculate energy change if dipole was flipped
    energy_diff = deltaU(state, i)

    #flip dipole if energy is same or lowered
    if energy_diff <= 0:
        state[i] = -state[i]
    else:
        if random.random() < (np.e)**(-energy_diff / temp):
            state[i] = -state[i]


state_final = np.copy(state)





# Convert values: 1 -> black (0 in colormap), -1 -> white (1 in colormap)
color_map1 = (state_initial == -1).astype(int).reshape(1, -1)
color_map2 = (state_final == -1).astype(int).reshape(1, -1)

# Create a figure with two subplots stacked vertically
fig, axes = plt.subplots(2, 1, figsize=(10, 4))  # 2 rows, 1 column

# Function to plot an array with grid
def plot_array(ax, color_map, title):
    ax.imshow(color_map, cmap='gray', vmin=0, vmax=1, aspect='auto')
    num_cols = color_map.shape[1]

    # Draw grid lines
    for x in range(num_cols + 1):
        ax.plot([x - 0.5, x - 0.5], [-0.5, 0.5], color='black', linewidth=2)
    ax.plot([-0.5, num_cols - 0.5], [-0.5, -0.5], color='black', linewidth=2)
    ax.plot([-0.5, num_cols - 0.5], [0.5, 0.5], color='black', linewidth=2)

    ax.set_xticks([])  # Remove tick marks
    ax.set_yticks([])
    ax.axis('off')  # Hide axes
    ax.set_title(title)

# Plot each array correctly
plot_array(axes[0], color_map1, "Array 1")  # First array
plot_array(axes[1], color_map2, "Array 2")  # Second array

plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()
