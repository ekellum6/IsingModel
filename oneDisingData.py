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

#function calculates energy for intial state, and creates an array to store it and future energies for generated states 
def energies(state, iterations):

    interaction_energy = np.zeros(state.size)
    for i in range(state.size):
        if i == (state.size -1):
            interaction_energy[i] = -1*state[i]*state[1]
        else:
            interaction_energy[i] = -1*state[i]*state[i+1]
    energy_array = np.zeros((state.size)*iterations + 1)
    energy_array[0] = np.sum(interaction_energy)

    return energy_array


#this creates an array for temerartures to be tested and sets the lattice size  
size = 100
T = np.arange(8,0,-0.1)      
average_energies = np.zeros(T.size)
average_energies_squared =np.zeros(T.size)
heat_capacity = np.zeros(T.size)
iteration_per_dipole = 2500
energy_iter=0

#generates initial random state
state = rand_array(size)

#main loop that interates a flips diploles one at a time 
for temp in T:

    energy_array = energies(state,iteration_per_dipole)

    for iter in range(size*iteration_per_dipole):
        #select random dipole 
        i = int(random.random()*(size))
        #calculate energy change if dipole was flipped
        energy_diff = deltaU(state, i)
        #flip dipole if energy is same or lowered
        if energy_diff <= 0:
            state[i] = -state[i]
            energy_array[iter+1] = (energy_array[iter] + energy_diff)
        else:
            if random.random() < (np.e)**(-energy_diff / temp):
                state[i] = -state[i]
                energy_array[iter+1] = (energy_array[iter] + energy_diff)
            else:
                energy_array[iter+1] = (energy_array[iter])
    

    average_energies[energy_iter] = np.sum(energy_array) / (energy_array.size)
    average_energies_squared[energy_iter] = np.sum(energy_array**2) / (energy_array.size)
    heat_capacity[energy_iter] = ((average_energies_squared[energy_iter] - average_energies[energy_iter]**2) / temp**2)
    energy_iter += 1


#plot average energy/per dipole versus temperature
plt.scatter(T, (average_energies / size))
plt.show()

#plot heat capacity/per dipole versus temperature/per dipole
plt.scatter(T,(heat_capacity / size))
plt.show()

