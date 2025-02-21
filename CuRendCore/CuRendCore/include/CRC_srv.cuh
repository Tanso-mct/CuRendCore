#pragma once

#include "CRC_config.h"
#include "CRC_container.h"
#include "CRC_resource.cuh"
#include "CRC_view.cuh"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>

class CRC_API ICRCShaderResourceView
{
public:
    virtual ~ICRCShaderResourceView() = default;
};

class CRC_API CRCShaderResourceView : public ICRCContainable, public ICRCView, public ICRCShaderResourceView
{
private:
    std::unique_ptr<ICRCResource>& resource;

public:
    ~CRCShaderResourceView() override = default;
    
    virtual std::unique_ptr<ICRCResource>& GetResource() override;
};

class CRC_API CRCID3D11ShaderResourceView : public ICRCContainable, public ICRCView, public ICRCShaderResourceView
{
private:
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> d3d11SRV;
    std::unique_ptr<ICRCResource> emptyResource = nullptr;

public:
    ~CRCID3D11ShaderResourceView() override = default;

    virtual std::unique_ptr<ICRCResource>& GetResource() override;
};