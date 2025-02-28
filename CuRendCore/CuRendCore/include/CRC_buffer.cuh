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
private:
    D3D11_BUFFER_DESC desc_ = {};
    D3D11_SUBRESOURCE_DATA initialData_ = {};

public:
    CRC_BUFFER_DESC(Microsoft::WRL::ComPtr<ID3D11Device>& device) : d3d11Device_(device) {}
    ~CRC_BUFFER_DESC() override = default;

    Microsoft::WRL::ComPtr<ID3D11Device>& d3d11Device_;

    D3D11_BUFFER_DESC& Desc() { return desc_; }
    D3D11_SUBRESOURCE_DATA& InitialData() { return initialData_; }

    UINT& ByteWidth() { return desc_.ByteWidth; }
    D3D11_USAGE& Usage() { return desc_.Usage; }
    UINT& BindFlags() { return desc_.BindFlags; }
    UINT& CPUAccessFlags() { return desc_.CPUAccessFlags; }
    UINT& MiscFlags() { return desc_.MiscFlags; }
    UINT& StructureByteStride() { return desc_.StructureByteStride; }

    const void* SysMem() { return initialData_.pSysMem; }
};

class CRC_API ICRCBuffer
{
public:
    virtual ~ICRCBuffer() = default;
    virtual const void GetDesc(D3D11_BUFFER_DESC* dst) = 0;

    virtual const UINT& GetByteWidth() const = 0;
};

class CRC_API CRCBufferFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCBufferFactoryL0_0() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCBuffer 
: public ICRCContainable, public ICRCCudaResource, public ICRCBuffer, public ICRCAllocatedMem
{
private:
    D3D11_BUFFER_DESC desc_ = {};

    void* memPtr_ = nullptr;
    UINT byteWidth_ = 0;

public:
    CRCBuffer();
    CRCBuffer(CRC_BUFFER_DESC& desc);
    virtual ~CRCBuffer() override;

    virtual HRESULT GetType(D3D11_RESOURCE_DIMENSION& type) override;
    virtual const void GetDesc(D3D11_BUFFER_DESC* dst) override;

    virtual void* const GetMem() override { return reinterpret_cast<void*>(memPtr_); }
    virtual void*& GetMemPtr() override { return reinterpret_cast<void*&>(memPtr_); }

    virtual const UINT& GetByteWidth() const override { return byteWidth_; }

    virtual void Malloc
    (
        UINT byteWidth, UINT pitch = 1, UINT slicePitch = 1,
        DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN
    ) override;
    virtual void Free() override;
};

class CRC_API CRCID3D11BufferFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCID3D11BufferFactoryL0_0() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCID3D11Buffer : public ICRCContainable, public ICRCD3D11Resource, public ICRCBuffer
{
private:
    Microsoft::WRL::ComPtr<ID3D11Buffer> d3d11Buffer_;

public:
    CRCID3D11Buffer() = default;
    virtual ~CRCID3D11Buffer() override = default;

    virtual Microsoft::WRL::ComPtr<ID3D11Resource>& GetResource();
    virtual Microsoft::WRL::ComPtr<ID3D11Buffer>& Get() { return d3d11Buffer_; }
    virtual HRESULT GetType(D3D11_RESOURCE_DIMENSION& type) override;

    virtual const void GetDesc(D3D11_BUFFER_DESC* dst) override;
    virtual const UINT& GetByteWidth() const override;
};