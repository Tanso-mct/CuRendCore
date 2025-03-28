#pragma once

#include "CuRendCore/include/config.h"
#include "CuRendCore/include/factory.h"
#include "CuRendCore/include/device_context.cuh"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CRC_BUFFER_DESC;
class CRC_TEXTURE2D_DESC;
class CRC_SHADER_RESOURCE_VIEW_DESC;
class CRC_RENDER_TARGET_VIEW_DESC;
class CRC_DEPTH_STENCIL_VIEW_DESC;

class CRC_API CRC_DEVICE_DESC : public IDESC
{
public:
    CRC_DEVICE_DESC(Microsoft::WRL::ComPtr<ID3D11Device>& device) : d3d11Device_(device) {}
    ~CRC_DEVICE_DESC() override = default;

    Microsoft::WRL::ComPtr<ID3D11Device>& d3d11Device_;

    CRC_FEATURE_LEVEL featureLevel_ = CRC_FEATURE_LEVEL::L0_0;
    CRC_RENDER_MODE renderMode_ = CRC_RENDER_MODE::CUDA;
};

class CRC_API ICRCDevice
{
public:
    virtual ~ICRCDevice() = default;
    virtual Microsoft::WRL::ComPtr<ID3D11Device>& GetD3D11Device() = 0;

    virtual HRESULT CreateBuffer(CRC_BUFFER_DESC& desc, std::unique_ptr<ICRCContainable>& buffer) = 0;
    virtual HRESULT CreateTexture2D(CRC_TEXTURE2D_DESC& desc, std::unique_ptr<ICRCContainable>& texture2d) = 0;

    virtual HRESULT CreateShaderResourceView
    (
        CRC_SHADER_RESOURCE_VIEW_DESC& desc,std::unique_ptr<ICRCContainable>& srv
    ) = 0;

    virtual HRESULT CreateRenderTargetView
    (
        CRC_RENDER_TARGET_VIEW_DESC& desc, std::unique_ptr<ICRCContainable>& rtv
    ) = 0;

    virtual HRESULT CreateDepthStencilView
    (
        CRC_DEPTH_STENCIL_VIEW_DESC& desc, std::unique_ptr<ICRCContainable>& dsv
    ) = 0;

    virtual std::unique_ptr<ICRCDeviceContext>& GetImmediateContext() = 0;
};

class CRC_API CRCDeviceFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCDeviceFactoryL0_0() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCDevice : public ICRCDevice, public ICRCContainable
{
private:
    Microsoft::WRL::ComPtr<ID3D11Device>& d3d11Device;

    std::unique_ptr<ICRCDeviceContext> immediateContext = nullptr;

    const std::unique_ptr<ICRCFactory> bufferFactory = nullptr;
    const std::unique_ptr<ICRCFactory> texture2DFactory = nullptr;

    const std::unique_ptr<ICRCFactory> srvFactory_ = nullptr;
    const std::unique_ptr<ICRCFactory> rtvFactory_ = nullptr;
    const std::unique_ptr<ICRCFactory> dsvFactory_ = nullptr;

public:
    CRCDevice
    (
        Microsoft::WRL::ComPtr<ID3D11Device>& d3d11Device,
        std::unique_ptr<ICRCDeviceContext> immediateContext,
        std::unique_ptr<ICRCFactory> bufferFactory, 
        std::unique_ptr<ICRCFactory> texture2DFactory,
        std::unique_ptr<ICRCFactory> srvFactory,
        std::unique_ptr<ICRCFactory> rtvFactory,
        std::unique_ptr<ICRCFactory> dsvFactory
    );

    ~CRCDevice() override;

    Microsoft::WRL::ComPtr<ID3D11Device>& GetD3D11Device() override { return d3d11Device; }

    HRESULT CreateBuffer(CRC_BUFFER_DESC& desc, std::unique_ptr<ICRCContainable>& buffer) override;
    HRESULT CreateTexture2D(CRC_TEXTURE2D_DESC& desc, std::unique_ptr<ICRCContainable>& texture2d) override;
    HRESULT CreateShaderResourceView(CRC_SHADER_RESOURCE_VIEW_DESC& desc,std::unique_ptr<ICRCContainable>& srv) override;
    HRESULT CreateRenderTargetView(CRC_RENDER_TARGET_VIEW_DESC& desc, std::unique_ptr<ICRCContainable>& rtv) override;
    HRESULT CreateDepthStencilView(CRC_DEPTH_STENCIL_VIEW_DESC& desc, std::unique_ptr<ICRCContainable>& dsv) override;

    std::unique_ptr<ICRCDeviceContext>& GetImmediateContext() override { return immediateContext; }
};