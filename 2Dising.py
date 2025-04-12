import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy import stats
from numba import jit,njit

savefigs = False

L = 100 #number of lattice points in each dimension
size = L*L #total number of lattice points
J = 1.0  #Coupling Coefficient
N = 5000 #number of monte-carlo steps for data collection
Neq = 5000 #number of MCS for equilibration
Nt = 100 #number of temperatures sampled
Temps = np.linspace(1,5.95,Nt)

Tc = 2.269185 #Known Value (https://theory.tifr.res.in/~tridib/ReferenceMaterial/PlischkeBergersen_Sec.6.1.pdf)

#Data Collection
M_t = np.zeros(N) #magnetization
E_t = np.zeros(N) #energy per spin/lattice point
E2_t = np.zeros(N) #energy^2 per spin/lattice point
s_T = []
M_T = np.zeros(Nt)
E_T = np.zeros(Nt)
Cv_T = np.zeros(Nt)


@njit
def MCS(state,expvals,L=L,size=size,J=J):
    randxy = np.random.randint(0,L,(size,2))
    for n_in in range(size):
        #choose a random lattice point
        xpos = randxy[n_in,0]
        xp = xpos+1
        if xp == L:
            xp = 0
        xm = xpos-1
        ypos = randxy[n_in,1]
        yp = ypos+1
        if yp == L:
            yp = 0
        ym = ypos-1
        #calculate change in energy of prospective flip
        dU = 2*J*state[xpos,ypos]*(state[xp,ypos] + state[xm,ypos] + state[xpos,yp] + state[xpos,ym])
        #decide whether to flip
        if dU <= 0:
            state[xpos,ypos] *= -1
        elif np.random.rand() < expvals[int(dU/4)]:
            state[xpos,ypos] *= -1
    return state

#Iterate over Temps
Tcounter = 0
for T in Temps:
    print(Tcounter)
    #Precompute exponentials
    expvals = [0.,np.exp(-4/T),np.exp(-8/T)]
    #Begin in Random Initial State
    state = np.random.choice([-1,1],(L,L))
    state_i = state.copy()
    #Begin Iteration of MCS (monte-carlo steps)
    for n_out in range(N+Neq):
        if n_out >= Neq:
            #compute and store magnetization
            M_t[n_out-Neq] = np.sum(state)/size
            #compute and store energy per spin
            E_t[n_out-Neq] = -J*np.sum(state*(np.roll(state,1,0)+np.roll(state,1,1)))/size
            #compute and store E^2 per spin
            E2_t[n_out-Neq] = E_t[n_out-Neq]**2
        #For each MCS, attempt flipping one spin for every lattice point
        MCS(state,expvals)
    M_T[Tcounter] = stats.mode(np.abs(M_t),axis=None)[0]
    #M_T[Tcounter] = np.mean(np.abs(M_t))
    E_T[Tcounter] = np.mean(E_t)
    Cv_T[Tcounter] = size*((np.sum(E2_t)/N)-(E_T[Tcounter]**2))/(T**2)
    state_f = state.copy()
    s_T.append((state_i,state_f))
    Tcounter += 1

#Plot <E/N>(T)

#Analytical Solution:
def analyticalE(T):
    k = (2*np.sinh(2*J/T))/(np.cosh(2*J/T))**2
    K = sp.ellipk(k**2)
    return (-J/np.tanh(2*J/T))*(1+(2/np.pi)*(-1+2*(np.tanh(2*J/T))**2)*K)
exactE_T = np.array([analyticalE(T) for T in Temps])

plt.plot(Temps,E_T,'.', label='Numerical Result')
plt.plot(Temps, exactE_T,'--', label='Analytical Result')
plt.title("Average Energy for 2-D Ising Model")
#plt.xscale('log')
plt.xlabel("Temperature (J/k)")
plt.xticks([1,2,Tc,3,4,5,6],[1,2,'Tc',3,4,5,6])
plt.axvline(x=Tc, color='k', linestyle='--', linewidth=1)
plt.ylabel("Average Energy per Spin")
plt.legend()
if savefigs:
    plt.savefig(f'Figures/Poster/2D/2D-Energy.jpg')
    plt.close()
else:
    plt.show()

#Plot Cv(T)/N

#Analytical Solution:
def analyticalCv(T):
    k = (2*np.sinh(2*J/T))/(np.cosh(2*J/T))**2
    K = sp.ellipk(k**2)
    Ek = sp.ellipe(k**2)
    return (4/np.pi)*(((J/T)/np.tanh(2*J/T))**2)*(K-Ek-(1-np.tanh(2*J/T)**2)*((np.pi/2)+(-1+2*np.tanh(2*J/T)**2)*K))
exactCv_T = np.array([analyticalCv(T) for T in Temps])

plt.plot(Temps,Cv_T,'.', label='Numerical Result')
plt.plot(Temps, exactCv_T,'--', label='Analytical Result')
plt.title("Specific Heat for 2-D Ising Model")
#plt.xscale('log')
plt.xlabel("Temperature (J/k)")
plt.ylabel("Specific Heat per Spin")
plt.xticks([1,2,Tc,3,4,5,6],[1,2,'Tc',3,4,5,6])
plt.axvline(x=Tc, color='k', linestyle='--', linewidth=1)
plt.legend()
if savefigs:
    plt.savefig(f'Figures/Poster/2D/2D-Cv.jpg')
    plt.close()
else:
    plt.show()

#Plot <M>(T)

#Analytical Solution:
def analyticalM(T,Tc=Tc):
    if T < Tc:
        return (1-(1-np.tanh(J/T)**2)**4/(16*np.tanh(J/T)**4))**(1/8)
    else:
        return 0
exactM_T = np.array([analyticalM(T) for T in Temps])

plt.plot(Temps,M_T,'.',label='Numerical Result')
plt.plot(Temps,exactM_T,'--',label='Analytical Result')
#plt.xscale('log')
plt.title("Magnetization for 2-D Ising Model")
plt.xlabel("Temperature (J/k)")
plt.ylabel("Magnetization per Spin")
plt.xticks([1,2,Tc,3,4,5,6],[1,2,'Tc',3,4,5,6])
plt.axvline(x=Tc, color='k', linestyle='--', linewidth=1)
if savefigs:
    plt.savefig(f'Figures/Poster/2D/2D-Mag.jpg')
    plt.close()
else:
    plt.show()

#Plot Spin Configurations
for i_T in range(Nt):
    print(f'Temperature: {Temps[i_T]}')
    f,ax = plt.subplots(1,2)
    ax[0].imshow(s_T[i_T][0],cmap='gray')
    ax[0].set_xticks([]);
    ax[0].set_yticks([]);
    ax[0].xaxis.set_label_position('bottom')
    ax[0].set_xlabel('Initial State',fontsize=16)
    ax[1].imshow(s_T[i_T][1],cmap='gray')
    ax[1].set_xticks([]);
    ax[1].set_yticks([]);   
    ax[1].xaxis.set_label_position('bottom')
    ax[1].set_xlabel('Final State',fontsize=16)
    f.suptitle(f'T = {Temps[i_T]:.2f}',y=.85,fontsize=16)
    plt.tight_layout()
    if savefigs:
        plt.savefig(f'Figures/Poster/2D/2D-spinconfigs/2Dconfig{i_T+1}.jpg')
        plt.close()
    else:
        plt.show()