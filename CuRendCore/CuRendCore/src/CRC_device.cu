#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_device.cuh"

#include "CRC_texture.cuh"
#include "CRC_buffer.cuh"

#include "CRC_srv.cuh"
#include "CRC_rtv.cuh"
#include "CRC_dsv.cuh"

std::unique_ptr<ICRCContainable> CRCDeviceFactoryL0_0::Create(IDESC &desc) const
{
    CRC_DEVICE_DESC* deviceDesc = CRC::As<CRC_DEVICE_DESC>(&desc);
    if (!deviceDesc)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to create device. Invalid description.");
#endif
        return nullptr;
    }

    std::unique_ptr<CRCDevice> device;
    if (deviceDesc->renderMode_ == CRC_RENDER_MODE::CUDA)
    {
        device = std::make_unique<CRCDevice>
        (
            deviceDesc->d3d11Device_,
            std::make_unique<CRCImmediateContext>(),
            std::make_unique<CRCBufferFactoryL0_0>(),
            std::make_unique<CRCTexture2DFactoryL0_0>(),
            std::make_unique<CRCShaderResourceViewFactoryL0_0>(),
            std::make_unique<CRCRenderTargetViewFactoryL0_0>(),
            std::make_unique<CRCDepthStencilViewFactoryL0_0>()
        );
    }
    else if (deviceDesc->renderMode_ == CRC_RENDER_MODE::D3D11)
    {
        ID3D11DeviceContext* d3d11DeviceContext = nullptr;
        deviceDesc->d3d11Device_->GetImmediateContext(&d3d11DeviceContext);
        if (!d3d11DeviceContext)
        {
#ifndef NDEBUG
            CRC::CoutError("Failed to create device. Failed to get immediate context.");
#endif
            throw std::runtime_error("Failed to create device. Failed to get immediate context.");
        }

        device = std::make_unique<CRCDevice>
        (
            deviceDesc->d3d11Device_,
            std::make_unique<CRCID3D11Context>(&d3d11DeviceContext),
            std::make_unique<CRCID3D11BufferFactoryL0_0>(),
            std::make_unique<CRCID3D11Texture2DFactoryL0_0>(),
            std::make_unique<CRCID3D11ShaderResourceViewFactoryL0_0>(),
            std::make_unique<CRCID3D11RenderTargetViewFactoryL0_0>(),
            std::make_unique<CRCID3D11DepthStencilViewFactoryL0_0>()
        );
    }
    else
    {
#ifndef NDEBUG
        CRC::CoutError("Failed to create device. Unknown render mode.");
#endif
        throw std::runtime_error("Failed to create device. Unknown render mode.");
    }

    return device;
}

CRCDevice::CRCDevice
(
    Microsoft::WRL::ComPtr<ID3D11Device> &d3d11Device, 
    std::unique_ptr<ICRCDeviceContext> immediateContext,
    std::unique_ptr<ICRCFactory> bufferFactory, 
    std::unique_ptr<ICRCFactory> texture2DFactory, 
    std::unique_ptr<ICRCFactory> srvFactory, 
    std::unique_ptr<ICRCFactory> rtvFactory, 
    std::unique_ptr<ICRCFactory> dsvFactory
): d3d11Device(d3d11Device)
, immediateContext(std::move(immediateContext))
, bufferFactory(std::move(bufferFactory))
, texture2DFactory(std::move(texture2DFactory))
, srvFactory_(std::move(srvFactory))
, rtvFactory_(std::move(rtvFactory))
, dsvFactory_(std::move(dsvFactory))
{
#ifndef NDEBUG
    CRC::Cout("CRC Device created.");
#endif
}

CRCDevice::~CRCDevice()
{
#ifndef NDEBUG
    CRC::Cout("CRC Device destroyed.");
#endif
}

HRESULT CRCDevice::CreateBuffer(CRC_BUFFER_DESC &desc, std::unique_ptr<ICRCContainable> &buffer)
{
    buffer = bufferFactory->Create(desc);
    if (!buffer) return E_FAIL;
    return S_OK;
}

HRESULT CRCDevice::CreateTexture2D(CRC_TEXTURE2D_DESC &desc, std::unique_ptr<ICRCContainable> &texture2d)
{
    texture2d = texture2DFactory->Create(desc);
    if (!texture2d) return E_FAIL;
    return S_OK;
}

HRESULT CRCDevice::CreateShaderResourceView(CRC_SHADER_RESOURCE_VIEW_DESC &desc, std::unique_ptr<ICRCContainable>& srv)
{
    srv = srvFactory_->Create(desc);
    if (!srv) return E_FAIL;
    return S_OK;
}

HRESULT CRCDevice::CreateRenderTargetView(CRC_RENDER_TARGET_VIEW_DESC &desc, std::unique_ptr<ICRCContainable> &rtv)
{
    rtv = rtvFactory_->Create(desc);
    if (!rtv) return E_FAIL;
    return S_OK;
}

HRESULT CRCDevice::CreateDepthStencilView(CRC_DEPTH_STENCIL_VIEW_DESC &desc, std::unique_ptr<ICRCContainable> &dsv)
{
    dsv = dsvFactory_->Create(desc);
    if (!dsv) return E_FAIL;
    return S_OK;
}
