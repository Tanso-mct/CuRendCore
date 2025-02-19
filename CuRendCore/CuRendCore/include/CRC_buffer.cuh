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
public:
    ~CRC_BUFFER_DESC() override = default;

    Microsoft::WRL::ComPtr<ID3D11Device>& device_;

    UINT byteWidth_;
    CRC_USAGE usage_;
    UINT bindFlags_;
    UINT cpuAccessFlags_;
    UINT miscFlags_;
    UINT structureByteStride_;

    CRC_SUBRESOURCE_DATA initialData_;
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
public:
    ~CRCBuffer() override = default;

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