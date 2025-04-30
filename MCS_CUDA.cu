// ising3d_gpu.cu
#include <cuda.h>
#include <curand_kernel.h>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <thread>
#include <mutex>

// -----------------------------
// Problem parameters
// -----------------------------
constexpr int L        = 100;                      // lattice length
constexpr int64_t Nspins = int64_t(L)*L*L;         // total spins
constexpr int  Neq      = 20000;                     // equilibration sweeps
constexpr int  Nmeas    = 20000;                     // measurement sweeps
constexpr int  Nt       = 500;                       // number of temperatures
constexpr double J      = 1.0;                      // coupling

// -----------------------------
// Device inline helpers
// -----------------------------
__device__ inline int idx3(int x, int y, int z) {
    return x + L*(y + L*z);
}

// initialize curand states, one per site
__global__ void setup_rng(curandState *rng, unsigned long seed) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= Nspins) return;
    curand_init(seed, idx, 0, &rng[idx]);
}

// randomly initialize spins +1 or -1
__global__ void init_spins(int8_t *state, curandState *rng) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= Nspins) return;
    float r = curand_uniform(&rng[idx]);
    state[idx] = (r < 0.5f ?  1 : -1);
}

// one Metropolis sweep on one sublattice (color=0 red, 1 black)
__global__ void sweep_kernel(
        int8_t *state,
        curandState *rng,
        const double *expval,
        int color
) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= Nspins) return;

    // compute 3D coords
    int z = idx / (L*L);
    int y = (idx / L) % L;
    int x = idx % L;
    // checkerboard mask
    if(((x+y+z)&1) != color) return;

    // neighbor coords w/ PBC
    int xp = (x+1==L?0:x+1), xm = (x==0?L-1:x-1);
    int yp = (y+1==L?0:y+1), ym = (y==0?L-1:y-1);
    int zp = (z+1==L?0:z+1), zm = (z==0?L-1:z-1);

    // sum of 6 neighbors
    int sum =
            state[idx3(xp,y, z)] + state[idx3(xm,y, z)] +
            state[idx3(x, yp,z)] + state[idx3(x, ym,z)] +
            state[idx3(x, y, zp)] + state[idx3(x, y, zm)];

    int8_t s = state[idx];
    double dU = 2.0 * J * s * sum;

    float r = curand_uniform(&rng[idx]);
    int bin = int(dU/4.0 + 0.5);       // maps 4->1,8->2,12->3
    if(dU <= 0.0 || r < expval[bin]) {
        state[idx] = -s;
    }
}

// measure total energy & magnetization for one configuration
// we atomically accumulate into energies[step] and mags[step]
__global__ void measure_kernel(
        int8_t *state,
        double *energies,
        double *mags,
        int step
) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= Nspins) return;

    // coords
    int z = idx / (L*L);
    int y = (idx / L) % L;
    int x = idx % L;

    // only count +x,+y,+z neighbor once per bond
    int xp = (x+1==L?0:x+1);
    int yp = (y+1==L?0:y+1);
    int zp = (z+1==L?0:z+1);

    double e_loc = -J * double(state[idx]) * (
            state[idx3(xp,y, z)] +
            state[idx3(x, yp,z)] +
            state[idx3(x, y, zp)]
    );
    double m_loc = double(state[idx]);

    // atomic adds into this step’s accumulator
    atomicAdd(&energies[step], e_loc);
    atomicAdd(&mags[step],    m_loc);
}

__global__ void measure_kernel_redux(
        int8_t    *state,
        double    *energies,
        double    *mags,
        int        step
) {
    extern __shared__ double sdata[];
    double *es = sdata;                  // sdata[0..blockDim.x-1]
    double *ms = sdata + blockDim.x;     // sdata[blockDim.x..2*blockDim.x-1]

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double e_loc = 0.0, m_loc = 0.0;
    if (idx < Nspins) {
        // recover (x,y,z)
        int z = idx / (L*L);
        int y = (idx / L) % L;
        int x = idx % L;
        // only +x,+y,+z so each bond is counted once
        int xp = (x+1==L?0:x+1),
                yp = (y+1==L?0:y+1),
                zp = (z+1==L?0:z+1);

        e_loc = -J * double(state[idx]) * (
                state[idx3(xp,y, z)] +
                state[idx3(x, yp,z)] +
                state[idx3(x, y, zp)]
        );
        m_loc = double(state[idx]);
    }

    // load into shared mem
    es[tid] = e_loc;
    ms[tid] = m_loc;
    __syncthreads();

    // reduction in shared memory
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            es[tid] += es[tid + s];
            ms[tid] += ms[tid + s];
        }
        __syncthreads();
    }

    // thread 0 does one atomic per block
    if (tid == 0) {
        atomicAdd(&energies[step], es[0]);
        atomicAdd(&mags[step],    ms[0]);
    }
}


// -----------------------------
// Host‐side driver
// -----------------------------
// Global mutex for synchronized console output
std::mutex console_mutex;

void run_on_device(int device_id, const std::vector<double>& temps_slice) {
    cudaSetDevice(device_id);

    // Allocate memory
    int8_t      *d_state;
    curandState *d_rng;
    double      *d_expval;
    double      *d_energies, *d_mags;

    cudaMalloc(&d_state,    Nspins*sizeof(int8_t));
    cudaMalloc(&d_rng,      Nspins*sizeof(curandState));
    cudaMalloc(&d_expval,   4*sizeof(double));
    cudaMalloc(&d_energies, Nmeas*sizeof(double));
    cudaMalloc(&d_mags,     Nmeas*sizeof(double));

    int threads = 256;
    int blocks  = (Nspins + threads - 1) / threads;
    setup_rng<<<blocks, threads>>>(d_rng, 1234u + device_id); // unique seed

    for (double T : temps_slice) {
        double beta = 1.0 / T;
        double h_exp[4] = {
                0.0,
                std::exp(-4.0 * beta),
                std::exp(-8.0 * beta),
                std::exp(-12.0 * beta)
        };
        cudaMemcpy(d_expval, h_exp, 4*sizeof(double), cudaMemcpyHostToDevice);
        init_spins<<<blocks, threads>>>(d_state, d_rng);

        for (int i = 0; i < Neq; ++i) {
            sweep_kernel<<<blocks, threads>>>(d_state, d_rng, d_expval, 0);
            sweep_kernel<<<blocks, threads>>>(d_state, d_rng, d_expval, 1);
        }

        cudaMemset(d_energies, 0, Nmeas*sizeof(double));
        cudaMemset(d_mags,     0, Nmeas*sizeof(double));

        for (int step = 0; step < Nmeas; ++step) {
            sweep_kernel<<<blocks, threads>>>(d_state, d_rng, d_expval, 0);
            sweep_kernel<<<blocks, threads>>>(d_state, d_rng, d_expval, 1);
            measure_kernel_redux<<<blocks, threads, 2*threads*sizeof(double)>>>(d_state, d_energies, d_mags, step);
        }

        std::vector<double> E(Nmeas), M(Nmeas);
        cudaMemcpy(E.data(), d_energies, Nmeas*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(M.data(), d_mags,     Nmeas*sizeof(double), cudaMemcpyDeviceToHost);

        // Stats + Print
        double sumE = 0, sumE2 = 0;
        std::unordered_map<int64_t, int> freq;
        for (int i = 0; i < Nmeas; ++i) {
            double eper = E[i] / double(Nspins);
            sumE  += eper;
            sumE2 += eper * eper;
            int64_t mabs = llabs(int64_t(std::round(M[i])));
            freq[mabs]++;
        }
        double meanEper = sumE / Nmeas;
        double Cv = double(Nspins)*(sumE2 / Nmeas - meanEper*meanEper)/(T*T);

        int64_t best_m=0; int best_c=0;
        for (auto& kv : freq) {
            if (kv.second > best_c) {
                best_c = kv.second;
                best_m = kv.first;
            }
        }
        double modeMper = double(best_m) / double(Nspins);

        std::lock_guard<std::mutex> lock(console_mutex);
        std::cout
                << device_id << ", "
                << T << ", "
                << meanEper << ", "
                << Cv       << ", "
                << modeMper << "\n" << std::flush;
    }

    cudaFree(d_state);
    cudaFree(d_rng);
    cudaFree(d_expval);
    cudaFree(d_energies);
    cudaFree(d_mags);
}

int main() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count < 1) {
        std::cerr << "No CUDA devices found.\n" << std::flush;
        return 1;
    }

    std::cout << "GPU ID, T, <E/N>, Cv, mode|M/N|\n"
              << std::fixed << std::setprecision(6) << std::flush;

    // Divide temperature range
    std::vector<double> Temps(Nt);
    for (int i = 0; i < Nt; ++i)
        Temps[i] = 0.1 + (8.0 - 0.1) * i / double(Nt - 1);

    std::vector<std::thread> threads;
    int temps_per_device = (Nt + device_count - 1) / device_count;
    for (int dev = 0; dev < device_count; ++dev) {
        int start = dev * temps_per_device;
        int end   = std::min(Nt, start + temps_per_device);
        if (start >= Nt) break;

        std::vector<double> slice(Temps.begin() + start, Temps.begin() + end);
        threads.emplace_back(run_on_device, dev, slice);
    }

    for (auto& t : threads) t.join();
    return 0;
}
