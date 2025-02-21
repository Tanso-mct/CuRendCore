#pragma once

#include "CRC_config.h"
#include "CRC_container.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CRC_API ICRCResource
{
public:
    virtual ~ICRCResource() = default;
};

class CRC_API ICRCCudaResource : public ICRCResource
{
public:
    virtual ~ICRCCudaResource() = default;

    virtual void* const GetMem() const = 0;
    virtual const UINT& GetByteWidth() const = 0;
    virtual const UINT& GetPitch() const = 0;
    virtual const UINT& GetSlicePitch() const = 0;
};

class CRC_API ICRCD3D11Resource : public ICRCResource
{
public:
    virtual ~ICRCD3D11Resource() = default;
    virtual Microsoft::WRL::ComPtr<ID3D11Resource>& GetResource() = 0;
};