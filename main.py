import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import norm
from scipy.special import ellipk

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


def theoretical_1D_magnetization_per_spin(T, J=1.0):
    """
    Theoretical average magnetization per spin for the 1D Ising model in zero external field.
    """
    return T*0


def theoretical_1D_critical_temp(J=1.0):
    """
    Returns the critical temperature Tc for the 1D Ising model.
    In 1D, there is no phase transition at any finite temperature.
    """
    return 0.0


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


def theoretical_2D_energy_per_spin(T, J=1.0):
    """
    Exact energy per spin for the 2D Ising model (infinite lattice, no external field).
    Uses Onsager's solution with elliptic integral.
    """
    if T <= 0:
        return -2 * J  # T → 0 limit
    beta = 1.0 / T
    x = 2 * J * beta
    sinh2x = np.sinh(x)
    cosh2x = np.cosh(x)
    tanh2x = np.tanh(x)
    coth2x = cosh2x / sinh2x
    k = 2 * sinh2x / cosh2x**2
    if k >= 1.0:
        return 0.0  # handle numerical issue just above Tc
    Kk = ellipk(k**2)
    return -J * coth2x * (1 + (2 / np.pi) * (2 * tanh2x**2 - 1) * Kk)


def theoretical_2D_specific_heat_per_spin(T, J=1.0, eps=1e-5):
    """
    Approximate specific heat per spin for the 2D Ising model by numerical derivative.
    """
    if T <= eps:
        return 0.0
    E_plus = theoretical_2D_energy_per_spin(T + eps, J)
    E_minus = theoretical_2D_energy_per_spin(T - eps, J)
    return (E_plus - E_minus) / (2 * eps)


def theoretical_2D_critical_temp(J=1.0):
    """
    Returns the exact critical temperature Tc for the 2D Ising model on a square lattice.

    Tc = 2J / ln(1 + sqrt(2))  [Onsager, 1944]
    """
    return 2 * J / np.log(1 + np.sqrt(2))


# ------------------ Simulation Over a Range of Temperatures ------------------ #

def run_1D_simulations_vs_temperature(temp_list, L=100, J=1.0, equilibration_steps=2000, n_steps=5000, seed=42):
    sim_energy = []
    sim_specific_heat = []
    sim_magnetization = []
    theo_energy = []
    theo_specific_heat = []
    theo_magnetization = []

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
        theo_magnetization.append(theoretical_1D_magnetization_per_spin(T, J))

        i += 1
        print((i / len(temp_list))*100)

    return (np.array(sim_energy), np.array(sim_specific_heat), np.array(sim_magnetization),
            np.array(theo_energy), np.array(theo_specific_heat), np.array(theo_magnetization))


def run_2D_simulations_vs_temperature(temp_list, L=20, J=1.0, equilibration_steps=5000, n_steps=5000, seed=42):
    sim_energy = []
    sim_specific_heat = []
    sim_magnetization = []
    theo_energy = []
    theo_specific_heat = []
    theo_magnetization = []

    i = 0
    for T in temp_list:
        # model = isingModel.IsingModel(L=L, dim=2, T=T, J=J, seed=seed)
        model = isingModel.GPUIsingModelOptimized(L=L, dim=2, T=T, J=J, seed=seed)

        energies, magnetizations = model.run(n_steps=n_steps, equilibration_steps=equilibration_steps)

        N = L * L  # Total number of spins in 2D

        avg_energy = compute_average_energy_per_spin(energies, N)
        specific_heat = compute_specific_heat(energies, T, N)
        avg_magnetization = compute_average_magnetization_per_spin(magnetizations, N)

        sim_energy.append(avg_energy)
        sim_specific_heat.append(specific_heat)
        sim_magnetization.append(avg_magnetization)

        theo_energy.append(theoretical_2D_energy_per_spin(T, J))
        theo_specific_heat.append(theoretical_2D_specific_heat_per_spin(T, J))
        theo_magnetization.append(theoretical_2D_magnetization_per_spin(T, J))

        i += 1
        print((i / len(temp_list)) * 100)

    return (np.array(sim_energy), np.array(sim_specific_heat), np.array(sim_magnetization),
            np.array(theo_energy), np.array(theo_specific_heat), np.array(theo_magnetization))


def run_3D_simulation_vs_temperatures(temp_list, L=20, n_steps=5000, equilibration_steps=5000, seed=42):
    sim_energy = []
    sim_specific_heat = []
    sim_magnetization = []

    for T in temp_list:
        model = isingModel.WolffIsingModel(L=L, dim=3, T=T, J=1.0, seed=seed)

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
    sim_energy, sim_specific_heat, sim_magnetization, theo_energy, theo_specific_heat, theo_magnetization = run_1D_simulations_vs_temperature(
        temp_list, L=100, J=1.0, equilibration_steps=5000, n_steps=5000, seed=42)

    # Estimate Tc
    Tc_exact = theoretical_1D_critical_temp()
    Tc_est = estimate_tc(temp_list, sim_specific_heat)
    print(f"Estimated critical temperature Tc ≈ {Tc_est:.3g}")

    _plot(temp_list, sim_energy, sim_specific_heat, sim_magnetization, theo_energy, theo_specific_heat, theo_magnetization, Tc_exact, Tc_est)


def plot_2D_results():
    # Define a range of temperatures around the critical region
    temp_list = np.linspace(1.5, 3.5, 20)

    sim_energy, sim_specific_heat, sim_magnetization, theo_energy, theo_specific_heat, theo_magnetization = run_2D_simulations_vs_temperature(
        temp_list, L=20, J=1.0, equilibration_steps=5000, n_steps=5000, seed=42)

    # Estimate Tc
    Tc_exact = theoretical_2D_critical_temp()
    Tc_est = estimate_tc(temp_list, sim_specific_heat)
    print(f"Estimated critical temperature Tc ≈ {Tc_est:.3g}")

    _plot(temp_list, sim_energy, sim_specific_heat, sim_magnetization, theo_energy, theo_specific_heat, theo_magnetization, Tc_exact, Tc_est)


def plot_3D_results():
    # Define a temperature range around the critical temperature (T_c ~ 4.5115)
    # temp_list = np.linspace(1.0, 8.0, 50)
    temp_list = generate_temperature_grid(4.5115, sigma=1.4)
    energy, heat, magnetization = run_3D_simulation_vs_temperatures(temp_list, L=30, n_steps=1000, equilibration_steps=1000, seed=42)

    Tc_est = estimate_tc(temp_list, heat)
    print(f"Estimated critical temperature Tc ≈ {Tc_est:.4f}")

    plt.figure(figsize=(15, 5))

    # Plot Energy per Spin vs Temperature
    plt.subplot(1, 3, 1)
    plt.plot(temp_list, energy, 'o-')
    plt.axvline(Tc_est, color='red', linestyle='--', label=f'Estimated Tc = {Tc_est:.3f}')
    plt.xlabel("Temperature T")
    plt.ylabel("Average Energy per Spin")
    plt.title("3D Ising Energy vs T")
    plt.legend()

    # Plot Specific Heat per Spin vs Temperature
    plt.subplot(1, 3, 2)
    plt.plot(temp_list, heat, 'o-')
    plt.axvline(Tc_est, color='red', linestyle='--', label=f'Estimated Tc = {Tc_est:.3f}')
    plt.xlabel("Temperature T")
    plt.ylabel("Specific Heat per Spin")
    plt.title("3D Ising Specific Heat vs T")
    plt.legend()

    # Plot Magnetization per Spin vs Temperature
    plt.subplot(1, 3, 3)
    plt.plot(temp_list, magnetization, 'o-')
    plt.axvline(Tc_est, color='red', linestyle='--', label=f'Estimated Tc = {Tc_est:.3f}')
    plt.xlabel("Temperature T")
    plt.ylabel("Magnetization per Spin")
    plt.title("3D Ising Magnetization vs T")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_ext_data_1d(file):
    data = np.loadtxt(file, delimiter=',', skiprows=1)
    temp_list = data[:, 0]
    sim_energy = data[:, 1]
    sim_specific_heat = data[:, 2]
    sim_magnetization = data[:, 3]

    theo_energy = []
    theo_specific_heat = []
    theo_magnetization = []

    for T in temp_list:
        theo_energy.append(theoretical_1D_energy_per_spin(T))
        theo_specific_heat.append(theoretical_1D_specific_heat_per_spin(T))
        theo_magnetization.append(theoretical_1D_magnetization_per_spin(T))

    # Estimate Tc
    Tc_exact = theoretical_1D_critical_temp()
    Tc_est = estimate_tc(temp_list, sim_specific_heat)

    _plot(temp_list, sim_energy, sim_specific_heat, sim_magnetization, theo_energy, theo_specific_heat, theo_magnetization, Tc_exact, Tc_est)


def plot_ext_data_2d(file):
    data = np.loadtxt(file, delimiter=',', skiprows=1)
    temp_list = data[:, 0]
    sim_energy = data[:, 1]
    sim_specific_heat = data[:, 2]
    sim_magnetization = data[:, 3]

    theo_energy = []
    theo_specific_heat = []
    theo_magnetization = []

    for T in temp_list:
        theo_energy.append(theoretical_2D_energy_per_spin(T))
        theo_specific_heat.append(theoretical_2D_specific_heat_per_spin(T))
        theo_magnetization.append(theoretical_2D_magnetization_per_spin(T))

    # Estimate Tc
    Tc_exact = theoretical_2D_critical_temp()
    Tc_est = estimate_tc(temp_list, sim_specific_heat)

    _plot(temp_list, sim_energy, sim_specific_heat, sim_magnetization, theo_energy, theo_specific_heat, theo_magnetization, Tc_exact, Tc_est)


def _plot(temp_list, sim_energy, sim_specific_heat, sim_magnetization, theo_energy, theo_specific_heat, theo_magnetization, Tc_exact, Tc_est):
    # Set consistent color scheme
    colors = ['C0', 'C1', 'C2', 'C3']  # Sim, Theo, Tc_est, Tc_exact

    # Create a figure with three subplots
    plt.figure(figsize=(15, 5))

    # Energy per spin
    plt.subplot(1, 3, 1)
    plt.plot(temp_list, sim_energy, 'o', label="Simulated Energy", color=colors[0])
    plt.plot(temp_list, theo_energy, '-', label="Theoretical Energy", color=colors[1])
    plt.axvline(Tc_est, linestyle='--', label=f'Estimated Tc = {Tc_est:.5g}', color=colors[2])
    plt.axvline(Tc_exact, linestyle='--', label=f'Exact Tc = {Tc_exact:.5g}', color=colors[3])
    plt.xlabel("Temperature T")
    plt.ylabel("Average Energy per Spin")
    plt.title(f'Energy vs Temperature \nMSE = {mean_squared_error(sim_energy, theo_energy):.3g}')
    plt.legend()

    # Specific heat per spin
    plt.subplot(1, 3, 2)
    plt.plot(temp_list, sim_specific_heat, 'o', label="Simulated Specific Heat", color=colors[0])
    plt.plot(temp_list, theo_specific_heat, '-', label="Theoretical Specific Heat", color=colors[1])
    plt.axvline(Tc_est, linestyle='--', label=f'Estimated Tc = {Tc_est:.5g}', color=colors[2])
    plt.axvline(Tc_exact, linestyle='--', label=f'Exact Tc = {Tc_exact:.5g}', color=colors[3])
    plt.xlabel("Temperature T")
    plt.ylabel("Specific Heat per Spin")
    plt.title(f'Specific Heat vs Temperature \nMSE = {mean_squared_error(sim_specific_heat, theo_specific_heat):.3g}')
    plt.legend()

    # Magnetization per spin
    plt.subplot(1, 3, 3)
    plt.plot(temp_list, sim_magnetization, 'o', label="Simulated Magnetization", color=colors[0])
    plt.plot(temp_list, theo_magnetization, '-', label="Theoretical Magnetization", color=colors[1])
    plt.axvline(Tc_est, linestyle='--', label=f'Estimated Tc = {Tc_est:.5g}', color=colors[2])
    plt.axvline(Tc_exact, linestyle='--', label=f'Exact Tc = {Tc_exact:.5g}', color=colors[3])
    plt.xlabel("Temperature T")
    plt.ylabel("Absolute Magnetization per Spin")
    plt.title(f'Magnetization vs Temperature \nMSE = {mean_squared_error(sim_magnetization, theo_magnetization):.3g}')
    plt.legend()

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


# ------------------ Other ------------------ #

def generate_temperature_grid(Tc, sigma=0.5, n_points=50, p_min=0.001, p_max=0.999):
    """
    Generate a deterministic grid of temperatures more densely spaced around Tc
    using the inverse CDF of a Gaussian.

    Parameters:
        Tc (float): Critical temperature around which to cluster points.
        sigma (float): Controls how tightly points cluster around Tc.
        n_points (int): Total number of temperature points.
        p_min (float): Minimum probability for the inverse CDF (avoid -inf).
        p_max (float): Maximum probability for the inverse CDF (avoid +inf).

    Returns:
        np.ndarray: Sorted array of temperature points.
    """
    p = np.linspace(p_min, p_max, n_points)
    return norm.ppf(p, loc=Tc, scale=sigma)


def estimate_tc(T, C):
    """
    Estimate critical temperature by finding the maximum of C(T).
    Uses a quadratic fit around the peak to refine the estimate.
    Returns the estimated Tc.
    """
    idx = np.argmax(C)
    Tc_peak = T[idx]
    # Quadratic refinement if possible
    if 0 < idx < len(T) - 1:
        xs = T[idx-1:idx+2]
        ys = C[idx-1:idx+2]
        coeff = np.polyfit(xs, ys, 2)  # a*x^2 + b*x + c
        a, b, _ = coeff
        Tc_quad = -b / (2 * a)
        return Tc_quad
    return Tc_peak


def mean_squared_error(sim, exact):
    return np.mean((sim - exact)**2)


if __name__ == '__main__':
    # import time
    # start = time.time()
    # plot_2D_results()
    # end = time.time()
    # print(end - start)

    plot_ext_data_1d('results_1d.csv')
    plot_ext_data_2d('results_2d_50k.csv')

    # plot_1D_results()
    # plot_2D_results()
    # plot_3D_results()
    # model = isingModel.GPUIsingModelOptimized(L=64, dim=1, T=0.1, J=1.0, seed=42)
    # animate_ising_1d(model, n_frames=500, steps_per_frame=1)

    # Use T ~ 4.5 (which corresponds to a critical temperature near T_c ~ 4.5115 for the 3D Ising model when J=1).
    # model = isingModel.GPUIsingModelOptimized(L=32, dim=3, T=1.0, J=1.0, seed=42)
    # animate_ising_3d(model, steps_per_frame=1, n_frames=500, save_movie=True)

    # animations.run_animate_ising_3d()
