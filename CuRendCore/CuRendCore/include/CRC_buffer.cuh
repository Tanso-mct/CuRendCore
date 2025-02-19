#pragma once

#include "CRC_config.h"
#include "CRC_resource.cuh"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CRC_API CRCIBuffer
{
public:
    virtual ~CRCIBuffer() = default;
    
};

class CRC_API CRCBuffer : public ICRCContainable, public ICRCResource, public CRCIBuffer
{
public:
    ~CRCBuffer() override = default;

    virtual void* GetMem() const override;
    virtual std::size_t GetSize() const override;
};

class CRC_API CRCID3D11Buffer : public ICRCContainable, public ICRCResource, public CRCIBuffer
{
private:
    Microsoft::WRL::ComPtr<ID3D11Buffer> d3d11Buffer;

public:
    ~CRCID3D11Buffer() override = default;

    virtual void* GetMem() const override;
    virtual std::size_t GetSize() const override;
};