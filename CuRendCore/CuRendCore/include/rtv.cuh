#pragma once

#include "CuRendCore/include/config.h"
#include "CuRendCore/include/container.h"
#include "CuRendCore/include/view.cuh"
#include "CuRendCore/include/factory.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>

class CRC_API CRC_RENDER_TARGET_VIEW_DESC : public IDESC
{
public:
    CRC_RENDER_TARGET_VIEW_DESC() = delete;
    CRC_RENDER_TARGET_VIEW_DESC
    (
        Microsoft::WRL::ComPtr<ID3D11Device>& device, std::unique_ptr<ICRCContainable>& resource
    ) : d3d11Device_(device), resource_(resource) {}
    ~CRC_RENDER_TARGET_VIEW_DESC() override = default;

    Microsoft::WRL::ComPtr<ID3D11Device>& d3d11Device_;

    D3D11_RENDER_TARGET_VIEW_DESC desc_ = {};
    std::unique_ptr<ICRCContainable>& resource_;
};

class CRC_API ICRCRenderTargetView
{
public:
    virtual ~ICRCRenderTargetView() = default;
    virtual const void GetDesc(D3D11_RENDER_TARGET_VIEW_DESC* dst) = 0;
};

class CRC_API CRCRenderTargetViewFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCRenderTargetViewFactoryL0_0() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCRenderTargetView 
: public ICRCContainable, public ICRCView, public ICRCRenderTargetView
{
private:
    D3D11_RENDER_TARGET_VIEW_DESC desc_ = {};
    std::unique_ptr<ICRCContainable>& resource_;

public:
    CRCRenderTargetView() = delete;
    CRCRenderTargetView(std::unique_ptr<ICRCContainable>& resource, D3D11_RENDER_TARGET_VIEW_DESC& desc);
    virtual ~CRCRenderTargetView() override;

    // ICRCView
    virtual std::unique_ptr<ICRCContainable>& GetResource() override { return resource_; }

    // ICRCRenderTargetView
    virtual const void GetDesc(D3D11_RENDER_TARGET_VIEW_DESC* dst) override;
};

class CRC_API CRCID3D11RenderTargetViewFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCID3D11RenderTargetViewFactoryL0_0() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCID3D11RenderTargetView 
: public ICRCContainable, public ICRCView, public ICRCRenderTargetView
{
private:
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> d3d11RTV_;
    std::unique_ptr<ICRCContainable> emptyResource_ = nullptr;

public:
    CRCID3D11RenderTargetView();
    virtual ~CRCID3D11RenderTargetView() override;

    // ICRCView
    virtual std::unique_ptr<ICRCContainable>& GetResource() override { return emptyResource_; }

    // ICRCRenderTargetView
    virtual const void GetDesc(D3D11_RENDER_TARGET_VIEW_DESC* dst) override;

    virtual Microsoft::WRL::ComPtr<ID3D11RenderTargetView>& Get() { return d3d11RTV_; }
};