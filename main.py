import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import isingModel
import animations


# ------------------ Observable Functions ------------------ #

def compute_average_energy_per_spin(energies, N):
    """Compute the average energy per spin."""
    return np.mean(energies) / N


def compute_specific_heat(energies, T, N):
    """
    Compute the specific heat per spin from energy fluctuations.

    C = (beta^2 / N) * (⟨E^2⟩ - ⟨E⟩^2)
    """
    beta = 1.0 / T
    mean_E = np.mean(energies)
    mean_E2 = np.mean(energies ** 2)
    return (beta ** 2 / N) * (mean_E2 - mean_E ** 2)


def compute_average_magnetization_per_spin(magnetizations, N):
    """Compute the average absolute magnetization per spin."""
    return np.mean(np.abs(magnetizations)) / N


# ------------------ Theoretical Expressions for Ising Model ------------------ #

def theoretical_1D_energy_per_spin(T, J=1.0):
    """
    Theoretical average energy per spin for the 1D Ising model in zero external field.

    E/N = -J * tanh(J/(T))
    """
    beta = 1.0 / T
    return -J * np.tanh(beta * J)


def theoretical_1D_specific_heat_per_spin(T, J=1.0):
    """
    Theoretical specific heat per spin for the 1D Ising model.

    C/N = (beta * J)^2 * sech^2(beta * J)
    """
    beta = 1.0 / T
    return (beta * J) ** 2 * (1 / np.cosh(beta * J) ** 2)


def theoretical_2D_magnetization_per_spin(T, J=1.0):
    """
    Returns the exact spontaneous magnetization per spin for the 2D Ising model (Onsager's solution)
    for T < Tc. For T >= Tc, the magnetization is zero.

    m(T) = [1 - sinh(2J/T)^(-4)]^(1/8)  for T < Tc, with Tc ~ 2.269 for J=1.
    """
    Tc = 2.269
    if T >= Tc:
        return 0.0
    else:
        return (1 - np.sinh(2 * J / T) ** (-4)) ** (1 / 8)


# ------------------ Simulation Over a Range of Temperatures ------------------ #

def run_1D_simulations_vs_temperature(temp_list, L=100, J=1.0, equilibration_steps=2000, n_steps=5000, seed=42):
    sim_energy = []
    sim_specific_heat = []
    sim_magnetization = []
    theo_energy = []
    theo_specific_heat = []

    i = 0
    for T in temp_list:
        model = isingModel.IsingModel(L=L, dim=1, T=T, J=J, seed=seed)
        # model = isingModel.GPUIsingModelOptimized(L=L, dim=1, T=T, J=J, seed=seed)
        energies, magnetizations = model.run(n_steps=n_steps, equilibration_steps=equilibration_steps)
        N = L  # total number of spins for 1D

        avg_energy = compute_average_energy_per_spin(energies, N)
        specific_heat = compute_specific_heat(energies, T, N)
        avg_magnetization = compute_average_magnetization_per_spin(magnetizations, N)

        sim_energy.append(avg_energy)
        sim_specific_heat.append(specific_heat)
        sim_magnetization.append(avg_magnetization)

        theo_energy.append(theoretical_1D_energy_per_spin(T, J))
        theo_specific_heat.append(theoretical_1D_specific_heat_per_spin(T, J))

        i += 1
        print((i / len(temp_list))*100)

    return (np.array(sim_energy), np.array(sim_specific_heat),
            np.array(sim_magnetization), np.array(theo_energy), np.array(theo_specific_heat))


def run_2D_simulations_vs_temperature(temp_list, L=20, J=1.0, equilibration_steps=5000, n_steps=5000, seed=42):
    sim_energy = []
    sim_specific_heat = []
    sim_magnetization = []
    theo_magnetization = []

    i = 0
    for T in temp_list:
        # model = isingModel.IsingModel(L=L, dim=2, T=T, J=J, seed=seed)
        model = isingModel.GPUIsingModelOptimized(L=L, dim=2, T=T, J=J, seed=seed)

        import time
        start = time.time()

        energies, magnetizations = model.run(n_steps=n_steps, equilibration_steps=equilibration_steps)

        end = time.time()
        print(end - start)

        N = L * L  # Total number of spins in 2D

        avg_energy = compute_average_energy_per_spin(energies, N)
        specific_heat = compute_specific_heat(energies, T, N)
        avg_magnetization = compute_average_magnetization_per_spin(magnetizations, N)

        sim_energy.append(avg_energy)
        sim_specific_heat.append(specific_heat)
        sim_magnetization.append(avg_magnetization)
        theo_magnetization.append(theoretical_2D_magnetization_per_spin(T, J))

        i += 1
        print((i / len(temp_list)) * 100)

    return (np.array(sim_energy), np.array(sim_specific_heat),
            np.array(sim_magnetization), np.array(theo_magnetization))


def run_3D_simulation_vs_temperatures(temp_list, L=20, n_steps=5000, equilibration_steps=5000, seed=42):
    sim_energy = []
    sim_specific_heat = []
    sim_magnetization = []

    for T in temp_list:
        model = isingModel.GPUIsingModelOptimized(L=L, dim=3, T=T, J=1.0, seed=seed)

        import time
        start = time.time()

        energies, magnetizations = model.run(n_steps=n_steps, equilibration_steps=equilibration_steps)

        end = time.time()
        print(end - start)

        N = L ** 3  # Total number of spins in 3D
        sim_energy.append(compute_average_energy_per_spin(energies, N))
        sim_specific_heat.append(compute_specific_heat(energies, T, N))
        sim_magnetization.append(compute_average_magnetization_per_spin(magnetizations, N))
        print(
            f"T = {T:.3f}, E/N = {sim_energy[-1]:.4f}, C = {sim_specific_heat[-1]:.4f}, |m|/N = {sim_magnetization[-1]:.4f}")

    return np.array(sim_energy), np.array(sim_specific_heat), np.array(sim_magnetization)


# ------------------ Plotting Function ------------------ #

def plot_1D_results():
    # Define a range of temperatures
    temp_list = np.linspace(0.5, 5.0, 20)

    # Run the simulations (for 1D, where we can compare with theory)
    sim_energy, sim_specific_heat, sim_magnetization, theo_energy, theo_specific_heat = run_1D_simulations_vs_temperature(
        temp_list, L=100, J=1.0, equilibration_steps=5000, n_steps=5000, seed=42)

    # Create a figure with three subplots
    plt.figure(figsize=(15, 5))

    # Energy per spin
    plt.subplot(1, 3, 1)
    plt.plot(temp_list, sim_energy, 'o-', label="Simulated Energy")
    plt.plot(temp_list, theo_energy, 's--', label="Theoretical Energy")
    plt.xlabel("Temperature T")
    plt.ylabel("Average Energy per Spin")
    plt.title("Energy vs Temperature")
    plt.legend()

    # Specific heat per spin
    plt.subplot(1, 3, 2)
    plt.plot(temp_list, sim_specific_heat, 'o-', label="Simulated Specific Heat")
    plt.plot(temp_list, theo_specific_heat, 's--', label="Theoretical Specific Heat")
    plt.xlabel("Temperature T")
    plt.ylabel("Specific Heat per Spin")
    plt.title("Specific Heat vs Temperature")
    plt.legend()

    # Magnetization per spin
    plt.subplot(1, 3, 3)
    plt.plot(temp_list, sim_magnetization, 'o-', label="Simulated |Magnetization|")
    plt.xlabel("Temperature T")
    plt.ylabel("Absolute Magnetization per Spin")
    plt.title("Magnetization vs Temperature")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_2D_results():
    # Define a range of temperatures around the critical region
    temp_list = np.linspace(1.5, 3.5, 20)

    sim_energy, sim_specific_heat, sim_magnetization, theo_magnetization = run_2D_simulations_vs_temperature(
        temp_list, L=20, J=1.0, equilibration_steps=5000, n_steps=5000, seed=42)

    plt.figure(figsize=(15, 5))

    # Energy per Spin vs Temperature
    plt.subplot(1, 3, 1)
    plt.plot(temp_list, sim_energy, 'o-')
    plt.xlabel("Temperature T")
    plt.ylabel("Average Energy per Spin")
    plt.title("2D Ising Energy vs Temperature")

    # Specific Heat per Spin vs Temperature
    plt.subplot(1, 3, 2)
    plt.plot(temp_list, sim_specific_heat, 'o-')
    plt.xlabel("Temperature T")
    plt.ylabel("Specific Heat per Spin")
    plt.title("2D Ising Specific Heat vs Temperature")

    # Magnetization per Spin vs Temperature (with theoretical curve)
    plt.subplot(1, 3, 3)
    plt.plot(temp_list, sim_magnetization, 'o-', label="Simulated Magnetization")
    plt.plot(temp_list, theo_magnetization, 's--', label="Theoretical Magnetization")
    plt.xlabel("Temperature T")
    plt.ylabel("Absolute Magnetization per Spin")
    plt.title("2D Ising Magnetization vs Temperature")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_3D_results():
    # Define a temperature range around the critical temperature (T_c ~ 4.5115)
    temp_list = np.linspace(3.5, 5.5, 21)
    energy, heat, magnetization = run_3D_simulation_vs_temperatures(temp_list, L=20, n_steps=5000,
                                                                   equilibration_steps=5000, seed=42)

    plt.figure(figsize=(15, 5))

    # Plot Energy per Spin vs Temperature
    plt.subplot(1, 3, 1)
    plt.plot(temp_list, energy, 'o-')
    plt.xlabel("Temperature T")
    plt.ylabel("Average Energy per Spin")
    plt.title("3D Ising Energy vs T")

    # Plot Specific Heat per Spin vs Temperature
    plt.subplot(1, 3, 2)
    plt.plot(temp_list, heat, 'o-')
    plt.xlabel("Temperature T")
    plt.ylabel("Specific Heat per Spin")
    plt.title("3D Ising Specific Heat vs T")

    # Plot Magnetization per Spin vs Temperature
    plt.subplot(1, 3, 3)
    plt.plot(temp_list, magnetization, 'o-')
    plt.xlabel("Temperature T")
    plt.ylabel("Magnetization per Spin")
    plt.title("3D Ising Magnetization vs T")

    plt.tight_layout()
    plt.show()


# ------------------ Animations (1d & 2d) ------------------ #

def animate_ising_2d(model, n_frames=100, steps_per_frame=10):
    """
    Animate the lattice equilibration process.

    Parameters:
      model: an instance of GPUIsingModelOptimizedND (or similar)
      n_frames (int): number of frames in the animation.
      steps_per_frame (int): number of Metropolis steps between frames.

    Returns:
      ani: the Matplotlib animation object.
    """
    fig, ax = plt.subplots()
    # Initial frame: convert GPU lattice to CPU
    img = ax.imshow(model.lattice.get(), cmap='bwr', interpolation='nearest')
    ax.set_title(f'T = {model.T:.2f}')

    def update(frame):
        for _ in range(steps_per_frame):
            model.metropolis_step()
        # Update image data with the new lattice configuration
        img.set_data(model.lattice.get())
        ax.set_title(f'T = {model.T:.2f}, Frame {frame}')
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
    ani.save("ising_animation_2d.mp4", fps=24)


def animate_ising_1d(model, n_frames=100, steps_per_frame=5):
    """
    Create a raw animation for a 1D Ising model equilibration using the provided GPU model.

    Parameters:
      model: an instance of GPUIsingModelOptimized with dim=1.
      n_frames (int): Number of frames in the animation.
      steps_per_frame (int): Number of Metropolis sweeps per frame.
    """
    fig, ax = plt.subplots()
    x = np.arange(model.shape[0])
    # Initial plot: get lattice from GPU and plot as a line.
    line, = ax.plot(x, model.lattice.get(), 'o-', lw=2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Site index")
    ax.set_ylabel("Spin")
    ax.set_title(f"T = {model.T:.2f}")

    def update(frame):
        # Perform several Metropolis sweeps before updating the frame.
        for _ in range(steps_per_frame):
            model.metropolis_step()
        # Update the line with the current lattice configuration.
        lattice_cpu = model.lattice.get()  # Transfer only one frame.
        line.set_ydata(lattice_cpu)
        ax.set_title(f"T = {model.T:.2f}, Frame {frame}")
        return [line]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
    ani.save("ising_animation_1d.mp4", fps=24)


# ------------------ Run Simulations and Generate Graphs ------------------ #
if __name__ == '__main__':
    # plot_1D_results()
    # plot_2D_results()
    # plot_3D_results()
    # model = isingModel.GPUIsingModelOptimized(L=64, dim=1, T=0.1, J=1.0, seed=42)
    # animate_ising_1d(model, n_frames=500, steps_per_frame=1)

    # Use T ~ 4.5 (which corresponds to a critical temperature near T_c ~ 4.5115 for the 3D Ising model when J=1).
    # model = isingModel.GPUIsingModelOptimized(L=32, dim=3, T=1.0, J=1.0, seed=42)
    # animate_ising_3d(model, steps_per_frame=1, n_frames=500, save_movie=True)

    animations.run_animate_ising_3d()