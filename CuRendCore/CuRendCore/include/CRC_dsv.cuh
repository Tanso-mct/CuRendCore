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

class CRC_API ICRCDepthStencilView
{
public:
    virtual ~ICRCDepthStencilView() = default;
};

class CRC_API CRCDepthStencilView : public ICRCContainable, public ICRCView, public ICRCDepthStencilView
{
private:
    std::unique_ptr<ICRCResource>& resource;

public:
    ~CRCDepthStencilView() override = default;
    
    virtual std::unique_ptr<ICRCResource>& GetResource() override;
};

class CRC_API CRCID3D11DepthStencilView : public ICRCContainable, public ICRCView, public ICRCDepthStencilView
{
private:
    Microsoft::WRL::ComPtr<ID3D11DepthStencilView> d3d11DSV;
    std::unique_ptr<ICRCResource> emptyResource = nullptr;

public:
    ~CRCID3D11DepthStencilView() override = default;

    virtual std::unique_ptr<ICRCResource>& GetResource() override;
};

