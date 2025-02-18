#pragma once

#include "CRC_config.h"
#include "CRC_resource.cuh"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CRCTexture2D : public ICRCResource
{
public:
    virtual ~CRCTexture2D() = default;
};

class CRCID3D11Texture2D : public ICRCResource
{
private:
    Microsoft::WRL::ComPtr<ID3D11Texture2D> d3d11Texture2D;

public:
    ~CRCID3D11Texture2D() override = default;
};
