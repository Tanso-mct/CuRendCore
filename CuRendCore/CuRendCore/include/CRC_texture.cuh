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

class CRC_API CRC_TEXTURE2D_DESC : public IDESC
{
private:
    D3D11_TEXTURE2D_DESC desc_ = {};
    D3D11_SUBRESOURCE_DATA initialData_ = {};

public:
    CRC_TEXTURE2D_DESC(Microsoft::WRL::ComPtr<ID3D11Device>& device) : device_(device) {}
    ~CRC_TEXTURE2D_DESC() override = default;

    Microsoft::WRL::ComPtr<ID3D11Device>& device_;

    D3D11_TEXTURE2D_DESC& Desc() { return desc_; }
    D3D11_SUBRESOURCE_DATA& InitialData() { return initialData_; }

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

    const void* SysMem() { return initialData_.pSysMem; }
};

class CRC_API ICRCTexture2D
{
public:
    virtual ~ICRCTexture2D() = default;

    virtual const UINT& GetByteWidth() const = 0;
    virtual const UINT& GetPitch() const = 0;
    virtual const UINT& GetSlicePitch() const = 0;
    virtual const void GetDesc(D3D11_TEXTURE2D_DESC* dst) = 0;
};

class CRC_API CRCTexture2DFactory : public ICRCFactory
{
public:
    ~CRCTexture2DFactory() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCTexture2D : public ICRCContainable, public ICRCCudaResource, public ICRCTexture2D
{
private:
    D3D11_TEXTURE2D_DESC desc_ = {};
    std::unique_ptr<ICRCMem> dMem = nullptr;

public:
    CRCTexture2D() = default;
    ~CRCTexture2D() override = default;

    virtual void* const GetMem() const override { return dMem.get(); }

    virtual const UINT& GetByteWidth() const override { return dMem->GetByteWidth(); }
    virtual const UINT& GetPitch() const override { return dMem->GetPitch(); }
    virtual const UINT& GetSlicePitch() const override { return dMem->GetSlicePitch(); }
    virtual const void GetDesc(D3D11_TEXTURE2D_DESC* dst) override;

    friend class CRCTexture2DFactory;
};

class CRC_API CRCID3D11Texture2DFactory : public ICRCFactory
{
public:
    ~CRCID3D11Texture2DFactory() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCID3D11Texture2D : public ICRCContainable, public ICRCD3D11Resource, public ICRCTexture2D
{
private:
    D3D11_TEXTURE2D_DESC desc_ = {};
    Microsoft::WRL::ComPtr<ID3D11Texture2D> d3d11Texture2D;

public:
    CRCID3D11Texture2D() = default;
    ~CRCID3D11Texture2D() override = default;

    virtual Microsoft::WRL::ComPtr<ID3D11Resource>& GetResource() override;

    virtual const UINT& GetByteWidth() const override;
    virtual const UINT& GetPitch() const override;
    virtual const UINT& GetSlicePitch() const override;
    virtual const void GetDesc(D3D11_TEXTURE2D_DESC* dst) override;

    friend class CRCID3D11Texture2DFactory;
};
