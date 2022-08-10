#pragma once

#include <vector>
#include <cudalibxt.h>
#include "../cuda-array.hpp"

void err(const char* msg, ...);
using dtype = typename double;

// Always having two fields at the end
// These store min and max dimensions
typedef std::vector<dtype> FrameTerms;
typedef std::vector<dtype> DataPoints;
typedef std::vector<unsigned int> Indicies;

constexpr dtype PI = dtype(3.1415926535897932384626433 * 1);
constexpr dtype TAU = dtype(3.1415926535897932384626433 * 2);
constexpr dtype QUART = dtype(3.1415926535897932384626433 * 0.5);
constexpr dtype INVSCORE = std::numeric_limits<dtype>::infinity();

constexpr int blocksize = 256;
    
std::vector<dtype> CudeFrameTrials(
    const CudaArray<dtype>& data,
    const CudaArray<size_t>& factors,
    const CudaArray<size_t>& offsets,
    const CudaArray<dtype>& variations,
    const CudaArray<dtype>& amp_scales,
    size_t runs);

inline dim3 KG(size_t jobs, size_t threads = blocksize)
{
    unsigned int grid = unsigned int(jobs / threads);
    return jobs * threads == grid ? grid : grid + 1;
}

inline dim3 KT(size_t threads = blocksize)
{
    return unsigned int(threads);
}

struct WaveFitCache
{
    size_t points;
    DataPoints data;
    DataPoints last;
    dtype* data_;
    dtype* last_;
    size_t runs;
    dtype* scores_;

};

WaveFitCache CudePrepareWaveFit(const DataPoints& data);

bool CheckCudaAndSynchronize();