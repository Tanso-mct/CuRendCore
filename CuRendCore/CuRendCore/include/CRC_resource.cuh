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
    BUFFER_CPU_R = 1 << 0,
    BUFFER_CPU_W = 1 << 1,
    BUFFER_GPU_R = 1 << 2,
    BUFFER_GPU_W = 1 << 3,

    TEXTURE2D_CPU_R = 1 << 4,
    TEXTURE2D_CPU_W = 1 << 5,
    TEXTURE2D_GPU_R = 1 << 6,
    TEXTURE2D_GPU_W = 1 << 7,
};

class CRC_API ICRCResource
{
public:
    virtual ~ICRCResource() = default;
    virtual HRESULT GetType(UINT& rcType) = 0;
};