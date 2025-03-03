#pragma once

#include "CRC_config.h"
#include "CRC_container.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

enum class CRC_API CRC_RESOURCE_TYPE : UINT
{
    UNKNOWN = 0,
    BUFFER_CPU_R,
    BUFFER_CPU_W,
    BUFFER_GPU_R,
    BUFFER_GPU_W,

    TEXTURE2D_CPU_R,
    TEXTURE2D_CPU_W,
    TEXTURE2D_GPU_R,
    TEXTURE2D_GPU_W,
};

class CRC_API ICRCResource
{
public:
    virtual ~ICRCResource() = default;
    virtual HRESULT GetType(UINT& rcType) = 0;
};

class CRC_API ICRCCudaResource : public ICRCResource
{
public:
    virtual ~ICRCCudaResource() = default;
};

class CRC_API ICRCD3D11Resource : public ICRCResource
{
public:
    virtual ~ICRCD3D11Resource() = default;
    virtual Microsoft::WRL::ComPtr<ID3D11Resource>& GetResource() = 0;
};