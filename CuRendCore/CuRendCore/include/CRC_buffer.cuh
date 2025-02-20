#pragma once

#include "CRC_config.h"
#include "CRC_resource.cuh"
#include "CRC_factory.h"

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
    ~CRC_BUFFER_DESC() override = default;

    Microsoft::WRL::ComPtr<ID3D11Device>& device_;

    D3D11_BUFFER_DESC& Desc() { return desc_; }
    D3D11_SUBRESOURCE_DATA& InitialData() { return initialData_; }

    UINT& ByteWidth() { return desc_.ByteWidth; }
    D3D11_USAGE& Usage() { return desc_.Usage; }
    UINT& BindFlags() { return desc_.BindFlags; }
    UINT& CPUAccessFlags() { return desc_.CPUAccessFlags; }
    UINT& MiscFlags() { return desc_.MiscFlags; }
    UINT& StructureByteStride() { return desc_.StructureByteStride; }

    const void* SysMem() { return initialData_.pSysMem; }
    UINT& SysMemPitch() { return initialData_.SysMemPitch; }
    UINT& SysMemSlicePitch() { return initialData_.SysMemSlicePitch; }
};

class CRC_API CRCIBuffer
{
public:
    virtual ~CRCIBuffer() = default;
};

class CRC_API CRCBufferFactory : public ICRCFactory
{
public:
    ~CRCBufferFactory() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCBuffer : public ICRCContainable, public ICRCResource, public CRCIBuffer
{
private:
    CRC_CUDA_MEMORY mem = {};

public:
    ~CRCBuffer() override;

    virtual void* GetMem() const override;
    virtual std::size_t GetSize() const override;

    friend class CRCBufferFactory;
};

class CRC_API CRCID3D11BufferFactory : public ICRCFactory
{
public:
    ~CRCID3D11BufferFactory() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCID3D11Buffer : public ICRCContainable, public ICRCResource, public CRCIBuffer
{
private:
    Microsoft::WRL::ComPtr<ID3D11Buffer> d3d11Buffer = nullptr;

public:
    ~CRCID3D11Buffer() override = default;

    virtual void* GetMem() const override;
    virtual std::size_t GetSize() const override;

    friend class CRCID3D11BufferFactory;
};