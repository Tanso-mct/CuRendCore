#pragma once

#include "CRC_config.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Windows.h>
#include <memory>

struct CRC_API CRCCudaMem
{
    CRCMem* host = nullptr;
    CRCMem* device = nullptr;
};

class CRC_API CRCMem
{
private:
    const std::size_t size_ = 0;
    const UINT pitch_ = 0;
    const UINT slicePitch_ = 0;
    void* ptr_ = nullptr;

public:
    CRCMem(const std::size_t& size, const UINT& pitch, const UINT& slicePitch)
    : size_(size), pitch_(pitch), slicePitch_(slicePitch) {}

    ~CRCMem() = default;

    __device__ __host__ const std::size_t& Size() const { return size_; }
    __device__ __host__ const UINT& Pitch() const { return pitch_; }
    __device__ __host__ const UINT& SlicePitch() const { return slicePitch_; }

    __device__ __host__ void*& Mem() { return ptr_; }
    __device__ __host__ const void* Mem() const { return ptr_; }
};

namespace CRC
{

CRC_API HRESULT MallocCudaMem(CRCCudaMem& mem, const std::size_t& size, const UINT& pitch, const UINT& slicePitch);
CRC_API HRESULT SetCudaMem(CRCCudaMem& mem, const void* initialData);
CRC_API HRESULT FreeCudaMem(CRCCudaMem& mem);

}