#pragma once

#include "CRC_config.h"
#include "CRC_container.h"
#include "CRC_view.cuh"
#include "CRC_factory.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>

class CRC_API CRC_DEPTH_STENCIL_VIEW_DESC : public IDESC
{
public:
    CRC_DEPTH_STENCIL_VIEW_DESC() = delete;
    CRC_DEPTH_STENCIL_VIEW_DESC(std::unique_ptr<ICRCContainable>& resource) : resource_(resource) {}
    ~CRC_DEPTH_STENCIL_VIEW_DESC() override = default;

    D3D11_DEPTH_STENCIL_VIEW_DESC desc_ = {};
    std::unique_ptr<ICRCContainable>& resource_;
};

class CRC_API ICRCDepthStencilView
{
public:
    virtual ~ICRCDepthStencilView() = default;
    virtual const void GetDesc(D3D11_DEPTH_STENCIL_VIEW_DESC* dst) = 0;
};

class CRC_API CRCDepthStencilViewFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCDepthStencilViewFactoryL0_0() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCDepthStencilView 
: public ICRCContainable, public ICRCView, public ICRCDepthStencilView
{
private:
    D3D11_DEPTH_STENCIL_VIEW_DESC desc_ = {};
    std::unique_ptr<ICRCContainable>& resource_;

public:
    CRCDepthStencilView() = delete;
    CRCDepthStencilView(std::unique_ptr<ICRCContainable>& resource, D3D11_DEPTH_STENCIL_VIEW_DESC& desc);
    virtual ~CRCDepthStencilView() override = default;
    
    // ICRCView
    virtual std::unique_ptr<ICRCContainable>& GetResource() override { return resource_; }

    // ICRCDepthStencilView
    virtual const void GetDesc(D3D11_DEPTH_STENCIL_VIEW_DESC* dst) override;
};

class CRC_API CRCID3D11DepthStencilView 
: public ICRCContainable, public ICRCView, public ICRCDepthStencilView
{
private:
    Microsoft::WRL::ComPtr<ID3D11DepthStencilView> d3d11DSV;
    std::unique_ptr<ICRCContainable> emptyResource = nullptr;

public:
    virtual ~CRCID3D11DepthStencilView() override = default;

    // ICRCView
    virtual std::unique_ptr<ICRCContainable>& GetResource() override { return emptyResource; }

    // ICRCDepthStencilView
    virtual const void GetDesc(D3D11_DEPTH_STENCIL_VIEW_DESC* dst) override;
};

