#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_resource.cuh"

CRC_API void CRC::MallocCudaMem(CRC_CUDA_MEMORY &mem, const std::size_t &size)
{
    mem.size_ = size;
    CRC::CheckCuda(cudaMalloc(&mem.ptr_, size));
}

CRC_API void CRC::SetCudaMem(CRC_CUDA_MEMORY &mem, const D3D11_SUBRESOURCE_DATA &initialData)
{
    CRC::CheckCuda(cudaMemcpy(mem.ptr_, initialData.pSysMem, mem.size_, cudaMemcpyHostToDevice));
}

CRC_API void CRC::FreeCudaMem(CRC_CUDA_MEMORY &mem)
{
    CRC::CheckCuda(cudaFree(mem.ptr_));
}
