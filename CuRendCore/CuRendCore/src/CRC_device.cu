#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_device.cuh"

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
            std::make_unique<CRCBufferFactoryL0_0>(),
            std::make_unique<CRCTexture2DFactoryL0_0>()
        );
    }
    else if (deviceDesc->renderMode_ == CRC_RENDER_MODE::D3D11)
    {
        device = std::make_unique<CRCDevice>
        (
            deviceDesc->d3d11Device_,
            std::make_unique<CRCID3D11BufferFactoryL0_0>(),
            std::make_unique<CRCID3D11Texture2DFactoryL0_0>()
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

HRESULT CRCDevice::CreateBuffer(CRC_BUFFER_DESC& desc, std::unique_ptr<ICRCContainable> &buffer)
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
