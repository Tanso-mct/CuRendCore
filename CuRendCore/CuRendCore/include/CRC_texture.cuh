#pragma once

#include "CRC_config.h"
#include "CRC_container.h"
#include "CRC_resource.cuh"
#include "CRC_factory.h"
#include "CRC_buffer.cuh"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CRC_API CRC_TEXTURE_DESC : public IDESC
{
private:
    D3D11_TEXTURE2D_DESC desc_ = {};

public:
    ~CRC_TEXTURE_DESC() override = default;

    Microsoft::WRL::ComPtr<ID3D11Device>& device_;

    D3D11_TEXTURE2D_DESC& Desc() { return desc_; }

    UINT& Width() { return desc_.Width; }
    UINT& Height() { return desc_.Height; }
    UINT& MipLevels() { return desc_.MipLevels; }
    UINT& ArraySize() { return desc_.ArraySize; }
    DXGI_FORMAT& Format() { return desc_.Format; }
    DXGI_SAMPLE_DESC& SampleDesc() { return desc_.SampleDesc; }
    D3D11_USAGE& Usage() { return desc_.Usage; }
    UINT& BindFlags() { return desc_.BindFlags; }
    UINT& CPUAccessFlags() { return desc_.CPUAccessFlags; }
    UINT& MiscFlags() { return desc_.MiscFlags; }
};

class CRC_API CRCTextureFactory : public ICRCFactory
{
public:
    ~CRCTextureFactory() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API ICRCTexture2D
{
public:
    virtual ~ICRCTexture2D() = default;
};

class CRC_API CRCTexture2D : public ICRCContainable, public ICRCResource, public ICRCTexture2D
{
public:
    ~CRCTexture2D() override = default;

    virtual void* GetMem() const override;
    virtual std::size_t GetSize() const override;
};

class CRC_API CRCID3D11Texture2D : public ICRCContainable, public ICRCResource, public ICRCTexture2D
{
private:
    Microsoft::WRL::ComPtr<ID3D11Texture2D> d3d11Texture2D;

public:
    ~CRCID3D11Texture2D() override = default;

    virtual void* GetMem() const override;
    virtual std::size_t GetSize() const override;
};
