#pragma once

#include "CuRendCore/include/config.h"
#include "CuRendCore/include/container.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

enum class CRC_API CRC_RESOURCE_TYPE : UINT
{
    UNKNOWN = 1 << 0,
    CRC_RESOURCE = 1 << 1,
    D3D11_RESOURCE = 1 << 2,

    BUFFER = 1 << 3,
    TEXTURE2D = 1 << 4,

    CPU_R = 1 << 5,
    CPU_W = 1 << 6,
    GPU_R = 1 << 7,
    GPU_W = 1 << 8,
};

class CRC_API ICRCResource
{
public:
    virtual ~ICRCResource() = default;
    virtual HRESULT GetResourceType(UINT& rcType) = 0;
};

class CRC_API ICRCID3D11Resource
{
public:
    virtual ~ICRCID3D11Resource() = default;
    virtual Microsoft::WRL::ComPtr<ID3D11Resource> GetResource() = 0;
};