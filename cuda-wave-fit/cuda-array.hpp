#pragma once

#include "inc/cuda-wave-fit.h"
#include "cuda-array.hpp"
#include <cudalibxt.h>

template <typename T>
class CudaArray
{
private:
	T* host_ptr = nullptr;
	// Do not allow to copy us around
	CudaArray(const CudaArray<T>&) {};
	CudaArray() {};
public:

	size_t size = 0;

	T* dev_ptr = nullptr;

	CudaArray(size_t size)
	{
		this->size = size; // Remember the size (must not be changed!)
		cudaMalloc(reinterpret_cast<void**>(&dev_ptr), size * sizeof(T));
		// printf("Allocated dev ptr %p\n", (void*)dev_ptr);
	}

	void ToDevice(const std::vector<T>& data) const
	{
		if (data.size() != size) throw new
			std::exception("Size mismatch");
		cudaMemcpy(dev_ptr, data.data(),
			data.size() * sizeof(T),
			cudaMemcpyHostToDevice);
	}

	CudaArray(const std::vector<T>& data)
		: CudaArray(data.size())
	{
		ToDevice(data);
	}

	std::vector<T> FromDevice() const
	{
		std::vector<T> data(size);
		cudaMemcpy(data.data(), dev_ptr,
			data.size() * sizeof(T),
			cudaMemcpyDeviceToHost);
		return data;
	}

	~CudaArray()
	{
		// printf("Free dev ptr %p\n", (void*)dev_ptr);
		cudaFree(dev_ptr);
		dev_ptr = nullptr;
	}

};
