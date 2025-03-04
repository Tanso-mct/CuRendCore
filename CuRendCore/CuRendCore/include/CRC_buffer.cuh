#pragma once

#include "CRC_config.h"
#include "CRC_resource.cuh"
#include "CRC_factory.h"
#include "CRC_memory.cuh"

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
    virtual void* const GetMem() = 0;
};

class CRC_API CRCBufferFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCBufferFactoryL0_0() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCBuffer 
: public ICRCContainable, public ICRCResource, public ICRCBuffer, public ICRCMemory
{
private:
    D3D11_BUFFER_DESC desc_ = {};
    UINT rcType_ = 0;

    void* memPtr_ = nullptr;
    UINT byteWidth_ = 0;

public:
    CRCBuffer() = delete;
    
    CRCBuffer(CRC_BUFFER_DESC& desc);
    virtual ~CRCBuffer() override;

    // ICRCResource
    virtual HRESULT GetType(UINT& rcType) override;

    // ICRCBuffer
    virtual const void GetDesc(D3D11_BUFFER_DESC* dst) override;
    virtual void* const GetMem() override { return reinterpret_cast<void*>(memPtr_); }

    // ICRCMemory
    virtual void Malloc(UINT byteWidth) override;
    virtual void Free() override;
};

class CRC_API CRCID3D11BufferFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCID3D11BufferFactoryL0_0() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCID3D11Buffer 
: public ICRCContainable, public ICRCResource, public ICRCBuffer
{
private:
    Microsoft::WRL::ComPtr<ID3D11Buffer> d3d11Buffer_;

public:
    CRCID3D11Buffer() = default;
    virtual ~CRCID3D11Buffer() override = default;

    // ICRCResource
    virtual HRESULT GetType(UINT& rcType) override;

    // ICRCBuffer
    virtual const void GetDesc(D3D11_BUFFER_DESC* dst) override;
    virtual void* const GetMem() override { return nullptr; }

    virtual Microsoft::WRL::ComPtr<ID3D11Buffer>& Get() { return d3d11Buffer_; }
};