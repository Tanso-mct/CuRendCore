#pragma once

#include "CRC_config.h"
#include "CRC_factory.h"
#include "CRC_texture.cuh"
#include "CRC_buffer.cuh"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

    const std::unique_ptr<ICRCFactory> bufferFactory = nullptr;
    const std::unique_ptr<ICRCFactory> texture2DFactory = nullptr;

public:
    CRCDevice
    (
        Microsoft::WRL::ComPtr<ID3D11Device>& d3d11Device,
        std::unique_ptr<ICRCFactory> bufferFactory, 
        std::unique_ptr<ICRCFactory> texture2DFactory
    )
    : d3d11Device(d3d11Device)
    , bufferFactory(std::move(bufferFactory))
    , texture2DFactory(std::move(texture2DFactory)) {}

    ~CRCDevice() override = default;

    Microsoft::WRL::ComPtr<ID3D11Device>& GetD3D11Device() override { return d3d11Device; }

    HRESULT CreateBuffer(CRC_BUFFER_DESC& desc, std::unique_ptr<ICRCContainable>& buffer) override;
    HRESULT CreateTexture2D(CRC_TEXTURE2D_DESC& desc, std::unique_ptr<ICRCContainable>& texture2d) override;
};