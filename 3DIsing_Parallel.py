import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy import stats
from numba import jit,njit
import multiprocessing as mp

savefigs = True

Tc = 1/0.221651 #Known Value (https://arxiv.org/pdf/2406.08531)

L = 100 #number of lattice points in each dimension
size = L*L*L #total number of lattice points
J = 1.0  #Coupling Coefficient
N = 1000 #number of monte-carlo steps for data collection
Neq = 1000 #number of MCS for equilibration
Nt = 100 #number of temperatures sampled

#Begin in Random Initial State
state0 = np.random.choice([-1,1],(L,L,L))

@njit
def MCS(state,expvals,L=L,size=size,J=J):
    randxyz = np.random.randint(0,L,(size,3))
    for n_in in range(size):
        #choose a random lattice point
        xpos = randxyz[n_in,0]
        xp = (xpos+1) % L
        xm = xpos-1
        ypos = randxyz[n_in,1]
        yp = (ypos+1) % L
        ym = ypos-1
        zpos = randxyz[n_in,2]
        zp = (zpos+1) % L
        zm = zpos-1
        #calculate change in energy of prospective flip
        dU = 2*J*state[xpos,ypos,zpos]*(state[xp,ypos,zpos] + state[xm,ypos,zpos] + state[xpos,yp,zpos] + state[xpos,ym,zpos] + state[xpos,ypos,zp] + state[xpos,ypos,zm])
        #decide whether to flip
        if dU <= 0:
            state[xpos,ypos,zpos] *= -1
        elif np.random.rand() < expvals[int(dU/4)]:
            state[xpos,ypos,zpos] *= -1

def IsingSim(T,MCS=MCS,state0=state0,L=L,size=size,J=J,N=N,Neq=Neq):
    #Precompute exponentials
    expvals = [0.,np.exp(-4/T),np.exp(-8/T),np.exp(-12/T)]
    #Create Copy of Initial State
    state = state0.copy()
    #Initialize Intermediate Observable Storage
    M_t = np.zeros(N) #magnetization
    E_t = np.zeros(N) #energy per spin/lattice point
    E2_t = np.zeros(N) #energy^2 per spin/lattice point
    #Begin Iteration of MCS (monte-carlo steps)
    for n_out in range(N+Neq):
        if n_out >= Neq:
            #compute and store magnetization per spin
            M_t[n_out-Neq] = np.sum(state)/size
            #compute and store energy per spin
            E_t[n_out-Neq] = -J*np.sum(state*(np.roll(state,1,0)+np.roll(state,1,1)+np.roll(state,1,2)))/size
            #compute and store E^2 per spin
            E2_t[n_out-Neq] = E_t[n_out-Neq]**2
        #For each MCS, attempt flipping one spin for every lattice point
        MCS(state,expvals)
    #Calculate and output observables
    M = stats.mode(np.abs(M_t),axis=None)[0]
    E = np.sum(E_t)/(N)
    Cv = size*((np.sum(E2_t)/N)-(E**2))/(T**2)
    return [M,E,Cv]

##########################################################

if __name__ == "__main__":
    Temps = np.linspace(3,6,Nt,endpoint=False)

    pool = mp.Pool(processes=50)
    result = pool.map(IsingSim,Temps)

    result = np.array(result)

    M_T = result[:,0]
    E_T = result[:,1]
    Cv_T = result[:,2]

    print(f'Critical Temperature = {Temps[np.argmax(Cv_T)]}')

    #Plot <E/N>(T)
    plt.plot(Temps,E_T)
    plt.title("Average Energy for 3-D Ising Model")
    plt.xlabel("Temperature (J/k)")
    plt.ylabel("Average Energy per Spin")
    if savefigs:
        plt.savefig(f'Figures/Paper/3D-Energy.jpg')
        plt.close()
    else:
        plt.show()

    #Plot Cv(T)/N
    plt.plot(Temps,Cv_T)
    plt.title("Specific Heat for 3-D Ising Model")
    plt.xlabel("Temperature (J/k)")
    plt.ylabel("Specific Heat per Spin")
    if savefigs:
        plt.savefig(f'Figures/Paper/3D-Cv.jpg')
        plt.close()
    else:
        plt.show()

    #Plot <M>(T)
    plt.plot(Temps,M_T)
    plt.title("Magnetization for 3-D Ising Model")
    plt.xlabel("Temperature (J/k)")
    plt.ylabel("Magnetization per Spin")
    if savefigs:
        plt.savefig(f'Figures/Paper/3D-Mag.jpg')
        plt.close()
    else:
        plt.show()