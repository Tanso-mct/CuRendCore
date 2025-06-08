#include "CRCBuffer/include/pch.h"
#include "CRCBuffer/include/buffer.cuh"

CRC::Buffer::Buffer(UINT cpuRWFlags, UINT gpuRWFlags, UINT size)
: 
type_(
    ((cpuRWFlags & (UINT)CRC::RW_FLAG::READ) ? (UINT)CRC::RESOURCE_TYPE::CPU_R : 0) |
    ((cpuRWFlags & (UINT)CRC::RW_FLAG::WRITE) ? (UINT)CRC::RESOURCE_TYPE::CPU_W : 0) |
    ((gpuRWFlags & (UINT)CRC::RW_FLAG::READ) ? (UINT)CRC::RESOURCE_TYPE::GPU_R : 0) |
    ((gpuRWFlags & (UINT)CRC::RW_FLAG::WRITE) ? (UINT)CRC::RESOURCE_TYPE::GPU_W : 0)),
size_(size)
{
    // Initialize the buffer with default values
    isValid_ = true;
    dData_ = nullptr;
    hData_ = nullptr;
}

CRC::Buffer::~Buffer()
{
    if (isValid_) Release();
}

HRESULT CRC::Buffer::Release()
{
    if (!isValid_)
    {
        CRCBuffer::CoutWrn({"Buffer is not valid.", "Buffer is already released."});
        return E_FAIL;
    }

    if (dData_)
    {
        CudaCore::Free(&dData_);
        dData_ = nullptr;
    }
    if (hData_)
    {
        CudaCore::FreeHost(&hData_);
        hData_ = nullptr;
    }

    isValid_ = false;

#ifndef NDEBUG
    CRCBuffer::CoutDebug({"Buffer released. Size: " + std::to_string(size_)});
#endif // NDEBUG

    return S_OK;
}

HRESULT CRC::Buffer::GetType(UINT &type) const
{
    if (!isValid_)
    {
        CRCBuffer::CoutWrn({"Buffer is not valid."});
        return E_FAIL;
    }

    type = type_;
    return S_OK;
}

void CRC::Buffer::GetDesc(IDesc *desc) const
{
}

HRESULT CRC::Buffer::GetSize(UINT &size) const
{
    if (!isValid_)
    {
        CRCBuffer::CoutWrn({"Buffer is not valid."});
        return E_FAIL;
    }

    size = size_;
    return S_OK;
}

HRESULT CRC::Buffer::GetDataDeviceSide(void **&data)
{
    if (!isValid_) return E_FAIL;

    data = &dData_;
    return S_OK;
}

HRESULT CRC::Buffer::GetDataHostSide(void **&data)
{
    if (!isValid_) return E_FAIL;

    data = &hData_;
    return S_OK;
}

CRC::BufferDesc::BufferDesc()
{
    // Initialize the buffer description with default values
    cpuRWFlags_ = 0;
    gpuRWFlags_ = 0;
    size_ = 0;
}

std::unique_ptr<CRC::IProduct> CRC::BufferFactory::Create(CRC::IDesc &desc) const
{
    CRC::BufferDesc* bufferDesc = WACore::As<CRC::BufferDesc>(&desc);
    if (!bufferDesc)
    {
        CRCBuffer::CoutWrn({"Failed to create buffer from desc.", "Desc is not CRC::BufferDesc."});
        return nullptr;
    }

    std::unique_ptr<CRC::IProduct> product = std::make_unique<CRC::Buffer>
    (
        bufferDesc->cpuRWFlags_, bufferDesc->gpuRWFlags_, bufferDesc->size_
    );

    {
        WACore::RevertCast<CRC::Buffer, CRC::IProduct> buffer(product);

        UINT type = 0;
        buffer()->GetType(type);

        UINT size = 0;
        buffer()->GetSize(size);

        if (type & (UINT)CRC::RESOURCE_TYPE::GPU_R || type & (UINT)CRC::RESOURCE_TYPE::GPU_W)
        {
            void** data = nullptr;
            buffer()->GetDataDeviceSide(data);

            CudaCore::Malloc(data, bufferDesc->size_);
        }

        if (type & (UINT)CRC::RESOURCE_TYPE::CPU_R || type & (UINT)CRC::RESOURCE_TYPE::CPU_W)
        {
            void** data = nullptr;
            buffer()->GetDataHostSide(data);

            CudaCore::MallocHost(data, bufferDesc->size_);
        }
    }

#ifndef NDEBUG
    CRCBuffer::CoutDebug({"Buffer created. Size: " + std::to_string(bufferDesc->size_)});
#endif // NDEBUG

    return product;
}