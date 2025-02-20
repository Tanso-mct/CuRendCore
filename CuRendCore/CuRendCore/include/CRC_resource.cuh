#pragma once

#include "CRC_config.h"
#include "CRC_container.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct CRC_CUDA_MEMORY
{
    void* ptr_ = nullptr;
    std::size_t size_ = 0;
    UINT pitch_ = 0;
    UINT slicePitch_ = 0;
};

namespace CRC
{

CRC_API void MallocCudaMem(CRC_CUDA_MEMORY& mem, const std::size_t& size);
CRC_API void SetCudaMem(CRC_CUDA_MEMORY& mem, const D3D11_SUBRESOURCE_DATA& initialData);
CRC_API void FreeCudaMem(CRC_CUDA_MEMORY& mem);

}

class CRC_API ICRCResource
{
public:
    virtual ~ICRCResource() = default;
    virtual void* GetMem() const = 0;
    virtual std::size_t GetSize() const = 0;
};