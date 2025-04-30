import cupy as cp
import numpy as np
from collections import deque

# ------------------ GPU-Accelerated Ising Model (Metropolis) ------------------ #
class GPUIsingModelOptimized:
    def __init__(self, L, dim=3, T=2.0, J=1.0, seed=None):
        """
        Initialize the optimized GPU-accelerated Ising model for n dimensions.

        Parameters:
            L (int): Linear size of the lattice (assumes a hypercube of size L^dim).
            dim (int): Dimension of the lattice.
            T (float): Temperature.
            J (float): Coupling constant.
            seed (int or None): Random seed for reproducibility.
        """
        self.L = L
        self.dim = dim
        self.T = T
        self.J = J
        self.beta = 1.0 / T
        self.shape = (L,) * dim
        if seed is not None:
            cp.random.seed(seed)
        # Initialize lattice with random spins (+1 or -1)
        self.lattice = cp.random.choice(cp.array([-1, 1]), size=self.shape)
        # Precompute a checkerboard mask: red if the sum of coordinates is even.
        indices = cp.indices(self.shape)  # shape: (dim, ...)
        self.red_mask = (cp.sum(indices, axis=0) % 2 == 0)
        self.black_mask = ~self.red_mask

    def compute_neighbor_sum(self):
        """
        Compute the sum of nearest neighbors for every site using periodic boundaries.
        """
        s = cp.zeros_like(self.lattice)
        for axis in range(self.dim):
            s += cp.roll(self.lattice, shift=1, axis=axis)
            s += cp.roll(self.lattice, shift=-1, axis=axis)
        return s

    def update_sublattice(self, mask):
        """
        Update all spins on the sublattice defined by 'mask' in parallel.
        """
        neighbor_sum = self.compute_neighbor_sum()
        # Compute the energy change for a flip at every site.
        delta_E = 2 * self.J * self.lattice * neighbor_sum
        # Restrict to the sites in the given sublattice.
        delta_E_masked = delta_E[mask]
        # Generate random numbers for all these sites.
        rand = cp.random.rand(delta_E_masked.size)
        # Metropolis acceptance: accept if delta_E < 0 or with probability exp(-beta*delta_E)
        accept = (delta_E_masked < 0) | (cp.exp(-self.beta * delta_E_masked) > rand)
        # Flip spins where accepted.
        spins = self.lattice[mask]
        spins[accept] = -spins[accept]
        self.lattice[mask] = spins

    def metropolis_step(self):
        """
        Perform one full Metropolis sweep: update red sublattice then black sublattice.
        """
        self.update_sublattice(self.red_mask)
        self.update_sublattice(self.black_mask)

    def total_energy(self):
        """
        Compute the total energy of the lattice.
        """
        E = 0
        for axis in range(self.dim):
            E += cp.sum(self.lattice * cp.roll(self.lattice, shift=-1, axis=axis))
        return -self.J * E

    def total_magnetization(self):
        """
        Compute the total magnetization.
        """
        return cp.sum(self.lattice)

    def run(self, n_steps, equilibration_steps=1000):
        """
        Run the simulation, minimizing memory transfers by preallocating measurement arrays.

        Parameters:
            n_steps (int): Number of measurement sweeps.
            equilibration_steps (int): Number of sweeps used for equilibration.

        Returns:
            energies (np.array): Array of total energy measurements.
            magnetizations (np.array): Array of total magnetization measurements.
        """
        # Equilibration phase
        for _ in range(equilibration_steps):
            self.metropolis_step()
        # Preallocate measurement arrays on GPU.
        energies_gpu = cp.empty(n_steps, dtype=cp.float64)
        magnetizations_gpu = cp.empty(n_steps, dtype=cp.float64)
        for i in range(n_steps):
            self.metropolis_step()
            energies_gpu[i] = self.total_energy()
            magnetizations_gpu[i] = self.total_magnetization()
        # Transfer measurements back to CPU once.
        return energies_gpu.get(), magnetizations_gpu.get()


# ------------------ CPU Ising Model (Metropolis) ------------------ #
class IsingModel:
    def __init__(self, L, dim=2, T=2.0, J=1.0, seed=None):
        """
        Initialize the Ising model.

        Parameters:
            L (int): Linear size of the lattice (assumes a hypercube of size L^dim).
            dim (int): Dimension of the lattice.
            T (float): Temperature.
            J (float): Coupling constant.
            seed (int or None): Random seed for reproducibility.
        """
        self.L = L
        self.dim = dim
        self.T = T
        self.J = J
        self.beta = 1.0 / T
        self.shape = (L,) * dim
        if seed is not None:
            np.random.seed(seed)
        # Initialize lattice with random spins: +1 or -1
        self.lattice = np.random.choice([-1, 1], size=self.shape)

    def total_energy(self):
        """
        Compute the total energy of the lattice using periodic boundary conditions.
        """
        E = 0
        for axis in range(self.dim):
            # np.roll shifts the array along the given axis (periodic boundaries)
            E += np.sum(self.lattice * np.roll(self.lattice, shift=-1, axis=axis))
        return -self.J * E

    def total_magnetization(self):
        """
        Compute the total magnetization of the lattice.
        """
        return np.sum(self.lattice)

    def local_neighbor_sum(self, idx):
        """
        Compute the sum of neighbor spins for a given lattice index.
        """
        s = 0
        for axis in range(self.dim):
            plus_idx = list(idx)
            minus_idx = list(idx)
            plus_idx[axis] = (plus_idx[axis] + 1) % self.L
            minus_idx[axis] = (minus_idx[axis] - 1) % self.L
            s += self.lattice[tuple(plus_idx)] + self.lattice[tuple(minus_idx)]
        return s

    def metropolis_step(self):
        """
        Perform one full Metropolis sweep (attempt to update every spin once in random order).
        """
        indices = list(np.ndindex(self.shape))
        np.random.shuffle(indices)
        for idx in indices:
            s = self.lattice[idx]
            neighbor_sum = self.local_neighbor_sum(idx)
            # Energy change if spin is flipped: ΔE = 2J s * (sum of neighbor spins)
            delta_E = 2 * self.J * s * neighbor_sum
            if delta_E < 0 or np.random.rand() < np.exp(-self.beta * delta_E):
                self.lattice[idx] = -s

    def run(self, n_steps, equilibration_steps=0):
        """
        Run the simulation for a number of sweeps.

        Returns:
            energies (np.array): Total energy measured at each sweep.
            magnetizations (np.array): Total magnetization measured at each sweep.
        """
        energies = []
        magnetizations = []

        # Equilibration phase
        for _ in range(equilibration_steps):
            self.metropolis_step()

        # Measurement phase
        for _ in range(n_steps):
            self.metropolis_step()
            energies.append(self.total_energy())
            magnetizations.append(self.total_magnetization())

        return np.array(energies), np.array(magnetizations)


# ------------------ CPU Ising Model (Wolff) ------------------ #
class WolffIsingModel:
    r"""
    Ising model with Wolff single–cluster updates.

    Public interface:
        • total_energy()
        • total_magnetization()
        • wolff_step() ⟶ one cluster flip
        • run(n_steps, equilibration_steps=0)
    """

    def __init__(self, L, dim=2, T=2.0, J=1.0, seed=None):
        self.L     = L
        self.dim   = dim
        self.T     = T
        self.J     = J
        self.beta  = 1.0 / T
        self.shape = (L,) * dim

        if seed is not None:
            np.random.seed(seed)

        # random ±1 spins
        self.lattice = np.random.choice([-1, 1], size=self.shape)

        # bond-formation probability  p = 1 - exp(-2βJ)
        self.p_add = 1.0 - np.exp(-2.0 * self.beta * self.J)

    def total_energy(self):
        E = 0
        for ax in range(self.dim):
            E += np.sum(self.lattice * np.roll(self.lattice, -1, axis=ax))
        return -self.J * E

    def total_magnetization(self):
        return np.sum(self.lattice)

    def _neighbors(self, site):
        """Generator yielding nearest-neighbour indices (periodic BC)."""
        for ax in range(self.dim):
            plus  = list(site); plus[ax]  = (plus[ax] + 1) % self.L
            minus = list(site); minus[ax] = (minus[ax] - 1) % self.L
            yield tuple(plus)
            yield tuple(minus)

    def wolff_step(self):
        """
        Build one cluster starting from a random seed site, then
        flip the entire cluster.  One call = one Monte-Carlo sweep.
        """
        seed_site = tuple(np.random.randint(0, self.L, size=self.dim))
        seed_spin = self.lattice[seed_site]

        # Breadth-first growth using a deque (≈ queue)
        cluster = set([seed_site])
        frontier = deque([seed_site])

        while frontier:
            site = frontier.popleft()
            for nbr in self._neighbors(site):
                if nbr in cluster:
                    continue
                if self.lattice[nbr] == seed_spin and np.random.rand() < self.p_add:
                    cluster.add(nbr)
                    frontier.append(nbr)

        # Flip the whole cluster
        for site in cluster:
            self.lattice[site] *= -1

    def run(self, n_steps, equilibration_steps=0):
        """
        Perform `equilibration_steps` cluster flips (discarded),
        then record energy & magnetisation after each of the next `n_steps`.
        """
        for _ in range(equilibration_steps):
            self.wolff_step()

        energies, mags = np.empty(n_steps), np.empty(n_steps)
        for i in range(n_steps):
            self.wolff_step()
            energies[i] = self.total_energy()
            mags[i]     = self.total_magnetization()
        return energies, mags
