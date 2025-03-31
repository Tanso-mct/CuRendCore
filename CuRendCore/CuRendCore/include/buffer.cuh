#pragma once

#include "CuRendCore/include/config.h"
#include "packages/WinAppCore/include/WACore.h"

#include "CuRendCore/include/resource.cuh"
#include "CuRendCore/include/factory.h"
#include "CuRendCore/include/memory.cuh"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CRC_API CRC_BUFFER_DESC : public IDESC
{
public:
    CRC_BUFFER_DESC(Microsoft::WRL::ComPtr<ID3D11Device>& device) : d3d11Device_(device) {}
    ~CRC_BUFFER_DESC() override = default;

    Microsoft::WRL::ComPtr<ID3D11Device>& d3d11Device_;

    D3D11_BUFFER_DESC desc_ = {};
    D3D11_SUBRESOURCE_DATA initialData_ = {};
};

class CRC_API ICRCBuffer
{
public:
    virtual ~ICRCBuffer() = default;
    virtual const void GetDesc(D3D11_BUFFER_DESC* dst) = 0;
    virtual void* const GetDevicePtr() = 0;
};

class CRC_API CRCBufferFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCBufferFactoryL0_0() override = default;
    virtual std::unique_ptr<WACore::IContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCBuffer 
: public WACore::IContainable, public ICRCResource, public ICRCBuffer, public ICRCMemory
{
private:
    D3D11_BUFFER_DESC desc_ = {};
    UINT resType_ = 0;

    void* dPtr_ = nullptr;
    void* hPtr_ = nullptr;
    UINT byteWidth_ = 0;

public:
    CRCBuffer() = delete;
    
    CRCBuffer(CRC_BUFFER_DESC& desc);
    virtual ~CRCBuffer() override;

    // ICRCResource
    virtual HRESULT GetResourceType(UINT& rcType) override;

    // ICRCBuffer
    virtual const void GetDesc(D3D11_BUFFER_DESC* dst) override;
    virtual void* const GetDevicePtr() override { return dPtr_; }

    // ICRCMemory
    virtual void Malloc(UINT byteWidth) override;
    virtual void Free() override;
    virtual void HostMalloc(UINT byteWidth) override;
    virtual void HostFree() override;

    virtual const UINT& GetByteWidth() override { return byteWidth_; }
    virtual const UINT& GetRowPitch() override { return 0; }
    virtual const UINT& GetDepthPitch() override { return 0; }
    virtual void* const GetHostPtr() override;

    virtual HRESULT SendHostToDevice() override;
    virtual HRESULT SendHostToDevice(const void *src, UINT srcByteWidth) override;
    virtual HRESULT SendDeviceToHost() override;

    virtual bool IsCpuAccessible() override;
};

class CRC_API CRCID3D11BufferFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCID3D11BufferFactoryL0_0() override = default;
    virtual std::unique_ptr<WACore::IContainable> Create(IDESC& desc) const override;
};

class CRC_API ICRCID3D11Buffer
{
public:
    virtual ~ICRCID3D11Buffer() = default;
    virtual Microsoft::WRL::ComPtr<ID3D11Buffer>& Get() = 0;
    virtual void SetResourceType(UINT& resType) = 0;
};

class CRC_API CRCID3D11Buffer 
: public WACore::IContainable, public ICRCResource, public ICRCID3D11Resource, public ICRCBuffer, public ICRCID3D11Buffer
{
private:
    Microsoft::WRL::ComPtr<ID3D11Buffer> d3d11Buffer_;
    UINT resType_ = 0;

public:
    virtual ~CRCID3D11Buffer() override;

    // ICRCResource
    virtual HRESULT GetResourceType(UINT& resType) override;

    // ICRCID3D11Resource
    virtual Microsoft::WRL::ComPtr<ID3D11Resource> GetResource() override;

    // ICRCBuffer
    virtual const void GetDesc(D3D11_BUFFER_DESC* dst) override;
    virtual void* const GetDevicePtr() override { return nullptr; }

    // ICRCID3D11Buffer
    virtual Microsoft::WRL::ComPtr<ID3D11Buffer>& Get() override { return d3d11Buffer_; }
    virtual void SetResourceType(UINT& resType) override { resType_ = resType; }
};