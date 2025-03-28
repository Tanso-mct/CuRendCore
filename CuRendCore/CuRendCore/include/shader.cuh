#pragma once

#include "CuRendCore/include/config.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CRC_SHADER_REGISTER void** registerBuffers, void** registerTextures, void** registerSamplers

enum class CRC_API CRC_SHADER_TYPE : UINT
{
    UNKNOWN = 1 << 0,
    CRC_SHADER = 1 << 1,
    D3D11_SHADER = 1 << 2,

    VERTEX = 1 << 3,
    PIXEL = 1 << 4,
};

CRC_API class ICRCShader
{
public:
    virtual ~ICRCShader() = default;
    virtual HRESULT GetShaderType(UINT& shaderType) = 0;
};