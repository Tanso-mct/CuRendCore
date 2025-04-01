#pragma once

#include "CuRendCore/include/config.h"
#include "CuRendCore/include/WinAppCore/WACore.h"

#include "CuRendCore/include/view.cuh"
#include "CuRendCore/include/factory.h"

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
    CRC_DEPTH_STENCIL_VIEW_DESC
    (
        Microsoft::WRL::ComPtr<ID3D11Device>& device, std::unique_ptr<WACore::IContainable>& resource
    ) : d3d11Device_(device), resource_(resource) {}
    ~CRC_DEPTH_STENCIL_VIEW_DESC() override = default;

    Microsoft::WRL::ComPtr<ID3D11Device>& d3d11Device_;

    D3D11_DEPTH_STENCIL_VIEW_DESC desc_ = {};
    std::unique_ptr<WACore::IContainable>& resource_;
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
    virtual std::unique_ptr<WACore::IContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCDepthStencilView 
: public WACore::IContainable, public ICRCView, public ICRCDepthStencilView
{
private:
    D3D11_DEPTH_STENCIL_VIEW_DESC desc_ = {};
    std::unique_ptr<WACore::IContainable>& resource_;

public:
    CRCDepthStencilView() = delete;
    CRCDepthStencilView(std::unique_ptr<WACore::IContainable>& resource, D3D11_DEPTH_STENCIL_VIEW_DESC& desc);
    virtual ~CRCDepthStencilView() override;
    
    // ICRCView
    virtual std::unique_ptr<WACore::IContainable>& GetResource() override { return resource_; }

    // ICRCDepthStencilView
    virtual const void GetDesc(D3D11_DEPTH_STENCIL_VIEW_DESC* dst) override;
};

class CRC_API CRCID3D11DepthStencilViewFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCID3D11DepthStencilViewFactoryL0_0() override = default;
    virtual std::unique_ptr<WACore::IContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCID3D11DepthStencilView 
: public WACore::IContainable, public ICRCView, public ICRCDepthStencilView
{
private:
    Microsoft::WRL::ComPtr<ID3D11DepthStencilView> d3d11DSV;
    std::unique_ptr<WACore::IContainable> emptyResource = nullptr;

public:
    virtual ~CRCID3D11DepthStencilView() override;

    // ICRCView
    virtual std::unique_ptr<WACore::IContainable>& GetResource() override { return emptyResource; }

    // ICRCDepthStencilView
    virtual const void GetDesc(D3D11_DEPTH_STENCIL_VIEW_DESC* dst) override;

    virtual Microsoft::WRL::ComPtr<ID3D11DepthStencilView>& Get() { return d3d11DSV; }
};

