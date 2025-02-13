#pragma once

#include "CRC_config.h"
#include "CRC_resource.cuh"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CRCBuffer : public ICRCResource
{
public:
    virtual ~CRCBuffer() = default;
};

class CRCID3D11Buffer : public ICRCResource
{
private:
    Microsoft::WRL::ComPtr<ID3D11Buffer> d3d11Buffer;

public:
    ~CRCID3D11Buffer() override = default;
};