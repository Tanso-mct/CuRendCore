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
public:
    CRC_TEXTURE2D_DESC(Microsoft::WRL::ComPtr<ID3D11Device>& device) : d3d11Device_(device) {}
    ~CRC_TEXTURE2D_DESC() override = default;

    Microsoft::WRL::ComPtr<ID3D11Device>& d3d11Device_;

    D3D11_TEXTURE2D_DESC desc_ = {};
    D3D11_SUBRESOURCE_DATA initialData_ = {};
};

class CRC_API ICRCTexture2D
{
public:
    virtual ~ICRCTexture2D() = default;
    virtual const void GetDesc(D3D11_TEXTURE2D_DESC* dst) = 0;
    virtual const UINT& GetByteWidth() { return 0; }
    virtual cudaArray* const GetCudaArray() { return nullptr; }
    virtual void* const GetHostPtr() { return nullptr; }

    virtual const cudaSurfaceObject_t& GetSurfaceObject() { return 0; }
    virtual const cudaTextureObject_t& GetTextureObject() { return 0; }
};

class CRC_API CRCTexture2DFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCTexture2DFactoryL0_0() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCTexture2D 
: public ICRCContainable, public ICRCResource, public ICRCTexture2D, public ICRCMemory
{
private:
    D3D11_TEXTURE2D_DESC desc_ = {};
    UINT rcType_ = 0;

    cudaArray* cudaArray_ = nullptr;
    void* hPtr_ = nullptr;
    cudaSurfaceObject_t surfaceObject_ = 0;
    cudaTextureObject_t textureObject_ = 0;
    UINT byteWidth_ = 0;

public:
    CRCTexture2D() = delete;

    CRCTexture2D(CRC_TEXTURE2D_DESC& desc);
    ~CRCTexture2D() override;

    // ICRCResource
    virtual HRESULT GetType(UINT& rcType) override;

    // ICRCTexture2D
    virtual const void GetDesc(D3D11_TEXTURE2D_DESC* dst) override;
    virtual const UINT& GetByteWidth() override { return byteWidth_; }
    virtual cudaArray* const GetCudaArray() override { return cudaArray_; }
    virtual void* const GetHostPtr() override { return hPtr_; }
    virtual const cudaSurfaceObject_t& GetSurfaceObject() override { return surfaceObject_; }
    virtual const cudaTextureObject_t& GetTextureObject() override { return textureObject_; }
    
    // ICRCMemory
    virtual void Malloc(UINT byteWidth) override;
    virtual void Free() override;
    virtual void HostMalloc(UINT byteWidth) override;
    virtual void HostFree() override;
};

class CRC_API CRCCudaResource 
: public ICRCContainable, public ICRCResource, public ICRCTexture2D, public ICRCMemory
{
private:
    D3D11_TEXTURE2D_DESC desc_ = {};
    UINT rcType_ = 0;

    cudaArray* cudaArray_ = nullptr;
    void* hPtr_ = nullptr;
    cudaSurfaceObject_t surfaceObject_ = 0;
    cudaTextureObject_t textureObject_ = 0;
    UINT byteWidth_ = 0;

public:
    CRCCudaResource() = delete;

    CRCCudaResource(D3D11_TEXTURE2D_DESC& desc);
    ~CRCCudaResource() override;

    // ICRCResource
    virtual HRESULT GetType(UINT& rcType) override;

    // ICRCTexture2D
    virtual const void GetDesc(D3D11_TEXTURE2D_DESC* dst) override;
    virtual const UINT& GetByteWidth() override { return byteWidth_; }
    virtual cudaArray* const GetCudaArray() override { return cudaArray_; }
    virtual void* const GetHostPtr() override { return hPtr_; }
    virtual const cudaSurfaceObject_t& GetSurfaceObject() override { return surfaceObject_; }
    virtual const cudaTextureObject_t& GetTextureObject() override { return textureObject_; }

    // ICRCMemory
    virtual void Assign(void* const mem, UINT byteWidth) override;
    virtual void Unassign() override;
    virtual void HostMalloc(UINT byteWidth) override;
    virtual void HostFree() override;
};

class CRC_API CRCID3D11Texture2DFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCID3D11Texture2DFactoryL0_0() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCID3D11Texture2D 
: public ICRCContainable, public ICRCResource, public ICRCTexture2D
{
private:
    Microsoft::WRL::ComPtr<ID3D11Texture2D> d3d11Texture2D_;

public:
    CRCID3D11Texture2D() = default;
    ~CRCID3D11Texture2D() override;

    // ICRCResource
    virtual HRESULT GetType(UINT& rcType) override;

    // ICRCTexture2D
    virtual const void GetDesc(D3D11_TEXTURE2D_DESC* dst) override;

    virtual Microsoft::WRL::ComPtr<ID3D11Texture2D>& Get() { return d3d11Texture2D_; }
};
