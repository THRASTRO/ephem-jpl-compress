#include <cmath>
#include <vector>
#include <cassert>
#include <numeric>
#include <atomic>
#include <string>
#include <cudalibxt.h>


#include "../inc/cuda-wave-fit.h"
#include "../cuda-array.hpp"

// size_t calculus = 0;

__device__ inline size_t GetIndex()
{
    return blockIdx.x * blocksize + threadIdx.x;
}

__global__ void kernel_init_array(
    dtype* arr, size_t len, dtype val)
{
    const size_t i = GetIndex();
    if (i >= len) return;
    arr[i] = val;
}

__device__ std::atomic<double> score;

__device__ __host__ void GetVariations(
    size_t* factors, size_t n_factors,
    size_t* offsets, dtype* variations,
    dtype* amp_scales, size_t id, dtype* terms)
{
    double scaler = 1.0; size_t p = 0, i = 0;
    //for (size_t i = 0; i < n_factors; i += 3)
    {
        p = id / factors[i + 0];
        terms[i + 0] = variations[p + offsets[i + 0]];
        scaler = amp_scales[p];
        id -= factors[i + 0] * p;
        p = id / factors[i + 1];
        terms[i + 1] = variations[p + offsets[i + 1]];
        id -= factors[i + 1] * p;
        p = id / factors[i + 2];
        terms[i + 2] = variations[p + offsets[i + 2]] * scaler;
        id -= factors[i + 2] * p;
    }
}

__global__ void kernel_frame_trials(
    dtype* data, size_t points,
    size_t* factors, size_t n_factors,
    size_t* offsets, size_t n_offsets,
    dtype* variations, size_t n_variations,
    dtype* scores, size_t n_scores,
    dtype* amp_scales, size_t max_runs)
{
    size_t idx = GetIndex();
    if (idx >= max_runs) return;

    double* terms = new double[n_factors];
    GetVariations(factors, n_factors,
        offsets, variations,
        amp_scales, idx, terms);

    double delta = 0.0;
    for (size_t p = 0; p < points; p += 1)
    {
        double sum = 0.0;
        double ts = double(p) / (points - 1);
        for (size_t t = 0; t < n_factors; t += 3)
        {
            sum += sin((ts * terms[t+0]
                + terms[t+1])) * terms[t+2];
        }
        delta += abs(data[p] - sum);
    }
    delete[] terms;

    scores[idx] = delta;
}


__global__ void kernel_best_terms(
    size_t* factors, size_t n_factors,
    size_t* offsets, dtype* variations,
    dtype* scores, size_t n_scores,
    dtype* amp_scales, dtype* terms)
{
    dtype score = INVSCORE;
    size_t idx = std::string::npos;
    for (size_t i = 0; i < n_scores; i += 1)
    {
        if (scores[i] > score) continue;
        score = scores[i];
        idx = i;
    }
    GetVariations(factors, n_factors,
        offsets, variations,
        amp_scales, idx, terms);
}

std::vector<dtype> CudeFrameTrials(
    const CudaArray<dtype>& data,
    const CudaArray<size_t>& factors,
    const CudaArray<size_t>& offsets,
    const CudaArray<dtype>& variations,
    const CudaArray<dtype>& amp_scales,
    size_t runs)
{

    // Temporary array to hold scores
    CudaArray<dtype> scores(runs);

    CheckCudaAndSynchronize();

    // Calculate scores for all variations
    kernel_frame_trials <<< KG(runs), KT() >>> (
        data.dev_ptr, data.size,
        factors.dev_ptr, factors.size,
        offsets.dev_ptr, offsets.size,
        variations.dev_ptr, variations.size,
        scores.dev_ptr, scores.size,
        amp_scales.dev_ptr, runs);

    CheckCudaAndSynchronize();

    // Array to hold selected best terms
    CudaArray<dtype> best(factors.size);

    CheckCudaAndSynchronize();

    // Fill best terms from best score
    kernel_best_terms <<< 1, 1 >>> (
        factors.dev_ptr, factors.size,
        offsets.dev_ptr, variations.dev_ptr,
        scores.dev_ptr, scores.size,
        amp_scales.dev_ptr, best.dev_ptr);

    CheckCudaAndSynchronize();

    // Get terms back from GPU
    return best.FromDevice();
}

// Code provided by NVIDIA - newer sdk versions seem to
// support atomicAdd for doubles natively. Therefore I
// have renamed the function to not clash in the future.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val == 0.0)
        return __longlong_as_double(old);
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

bool CheckCudaAndSynchronize()
{
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cuda failed with error code %d (%s)\n",
            cudaStatus, cudaGetErrorString(cudaStatus));
        throw new std::exception("Cuda Execution error");
        return false;
    }
    cudaError_t deviceStatus = cudaDeviceSynchronize();
    if (deviceStatus != cudaSuccess) {
        fprintf(stderr, "Cuda sync failed with error code %d (%s)\n",
            deviceStatus, cudaGetErrorString(cudaStatus));
        throw new std::exception("Cuda Synchronize error");
        return false;
    }
    return true;
}

void CudaResetWaveFit(WaveFitCache caching)
{
    kernel_init_array <<< KG(caching.points), KT() >>>
        (caching.last_, caching.points, 0.0);
    CheckCudaAndSynchronize();
}

// Upload data to GPU and initialize memory for results
WaveFitCache CudePrepareWaveFit(const DataPoints& data)
{

    // auto runs = samples * samples * samples;

    WaveFitCache caching{};
    // caching.last = new dtype[data.size()];
    caching.points = data.size();
    // caching.data = data;
    // caching.runs = runs;

    // Create memory structure to hold the data for the wave fit run
    cudaMalloc(reinterpret_cast<void**>(&caching.data_), data.size() * sizeof(dtype));
    cudaMalloc(reinterpret_cast<void**>(&caching.last_), data.size() * sizeof(dtype));
    // cudaMalloc(reinterpret_cast<void**>(&caching.scores_), runs * sizeof(dtype));

    // Copy the data points over to the GPU so it can calculate the scores
    cudaMemcpy(caching.data_, data.data(), data.size() * sizeof(dtype), cudaMemcpyHostToDevice);

    // Reset `last` array to all zero on the GPU
    kernel_init_array <<< KG(caching.points), KT() >>>
        (caching.last_, caching.points, 0.0);
    // Reset `scores` array to all infinity on the GPU
    kernel_init_array <<< KG(caching.runs), KT() >>>
        (caching.scores_, caching.runs, INVSCORE);

    // Wait for all operations to finish
    CheckCudaAndSynchronize();

    return caching;
}
