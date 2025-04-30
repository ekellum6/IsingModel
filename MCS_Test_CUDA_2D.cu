// ising2d_gpu.cu
#include <cuda.h>
#include <curand_kernel.h>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <iomanip>

// -----------------------------
// Problem parameters
// -----------------------------
constexpr int L        = 100;
constexpr int64_t Nspins = int64_t(L)*L;
constexpr int  Neq      = 20000;
constexpr int  Nmeas    = 20000;
constexpr int  Nt       = 500;
constexpr double J      = 1.0;

// -----------------------------
// Device inline helpers
// -----------------------------
__device__ inline int idx2(int x, int y) {
    return x + L*y;
}

__global__ void setup_rng(curandState *rng, unsigned long seed) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= Nspins) return;
    curand_init(seed, idx, 0, &rng[idx]);
}

__global__ void init_spins(int8_t *state, curandState *rng) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= Nspins) return;
    float r = curand_uniform(&rng[idx]);
    state[idx] = (r < 0.5f ?  1 : -1);
}

__global__ void sweep_kernel(
        int8_t *state,
        curandState *rng,
        const double *expval,
        int color
) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= Nspins) return;

    int y = idx / L;
    int x = idx % L;
    if(((x + y) & 1) != color) return;

    int xp = (x + 1) % L, xm = (x + L - 1) % L;
    int yp = (y + 1) % L, ym = (y + L - 1) % L;

    int sum =
            state[idx2(xp, y)] + state[idx2(xm, y)] +
            state[idx2(x, yp)] + state[idx2(x, ym)];

    int8_t s = state[idx];
    double dU = 2.0 * J * s * sum;

    float r = curand_uniform(&rng[idx]);
    int bin = int(dU / 4.0 + 0.5);
    if(dU <= 0.0 || r < expval[bin]) {
        state[idx] = -s;
    }
}

__global__ void measure_kernel_redux(
        int8_t    *state,
        double    *energies,
        double    *mags,
        int        step
) {
    extern __shared__ double sdata[];
    double *es = sdata;
    double *ms = sdata + blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double e_loc = 0.0, m_loc = 0.0;
    if (idx < Nspins) {
        int y = idx / L;
        int x = idx % L;

        int xp = (x + 1) % L;
        int yp = (y + 1) % L;

        e_loc = -J * double(state[idx]) * (
                state[idx2(xp, y)] +
                state[idx2(x, yp)]
        );
        m_loc = double(state[idx]);
    }

    es[tid] = e_loc;
    ms[tid] = m_loc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            es[tid] += es[tid + s];
            ms[tid] += ms[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&energies[step], es[0]);
        atomicAdd(&mags[step], ms[0]);
    }
}

int main(){
    std::cout << "Starting 2D Ising Model CUDA\n" << std::flush;

    std::vector<double> Temps(Nt);
    for (int i = 0; i < Nt; ++i)
        Temps[i] = 0.1 + (8.0 - 0.1) * i / double(Nt - 1);

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
    int blocks  = (Nspins + threads-1)/threads;
    setup_rng<<<blocks,threads>>>(d_rng, 1234u);

    std::cout << "T, <E/N>, Cv, mode|M/N|\n"
              << std::fixed << std::setprecision(6) << std::flush;

    for(int ti=0; ti<Nt; ++ti){
        double T = Temps[ti];
        double beta = 1.0/T;
        double h_exp[4] = {0.0,
                           std::exp(-4.0*beta),
                           std::exp(-8.0*beta),
                           std::exp(-12.0*beta)
        };
        cudaMemcpy(d_expval, h_exp, 4*sizeof(double), cudaMemcpyHostToDevice);
        init_spins<<<blocks,threads>>>(d_state, d_rng);

        for(int i=0;i<Neq;++i){
            sweep_kernel<<<blocks,threads>>>(d_state,d_rng,d_expval,0);
            sweep_kernel<<<blocks,threads>>>(d_state,d_rng,d_expval,1);
        }

        cudaMemset(d_energies, 0, Nmeas*sizeof(double));
        cudaMemset(d_mags,     0, Nmeas*sizeof(double));

        for(int step=0; step<Nmeas; ++step){
            sweep_kernel<<<blocks,threads>>>(d_state,d_rng,d_expval,0);
            sweep_kernel<<<blocks,threads>>>(d_state,d_rng,d_expval,1);
            measure_kernel_redux<<<blocks,threads, 2*threads*sizeof(double)>>>(d_state, d_energies, d_mags, step);
        }

        std::vector<double> E(Nmeas), M(Nmeas);
        cudaMemcpy(E.data(), d_energies, Nmeas*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(M.data(), d_mags,     Nmeas*sizeof(double), cudaMemcpyDeviceToHost);

        double sumE = 0, sumE2 = 0;
        std::unordered_map<int64_t,int> freq;
        for(int i=0;i<Nmeas;++i){
            double eper = E[i]/double(Nspins);
            sumE  += eper;
            sumE2 += eper*eper;
            int64_t mabs = llabs(int64_t(std::round(M[i])));
            freq[mabs]++;
        }
        double meanEper = sumE / Nmeas;
        double Cv = double(Nspins)*(sumE2/Nmeas - meanEper*meanEper)/(T*T);

        int64_t best_m=0; int best_c=0;
        for(auto &kv: freq){
            if(kv.second>best_c){
                best_c = kv.second;
                best_m = kv.first;
            }
        }
        double modeMper = double(best_m)/double(Nspins);

        std::cout
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
    return 0;
}
