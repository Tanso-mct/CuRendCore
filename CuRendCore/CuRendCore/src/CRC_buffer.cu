#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_buffer.cuh"

std::unique_ptr<ICRCContainable> CRCBufferFactory::Create(IDESC &desc) const
{
    CRC_BUFFER_DESC* bufferDesc = CRC::As<CRC_BUFFER_DESC>(&desc);
    if (!bufferDesc) return nullptr;

    std::unique_ptr<CRCBuffer> buffer = std::make_unique<CRCBuffer>();

    CRC::MallocCudaMem(buffer->mem, bufferDesc->Desc().ByteWidth);
    CRC::SetCudaMem(buffer->mem, bufferDesc->InitialData());

    return buffer;
}

CRCBuffer::~CRCBuffer()
{
    CRC::FreeCudaMem(mem);
}

void *CRCBuffer::GetMem() const
{
    return nullptr;
}

std::size_t CRCBuffer::GetSize() const
{
    return 0;
}

std::unique_ptr<ICRCContainable> CRCID3D11BufferFactory::Create(IDESC &desc) const
{
    CRC_BUFFER_DESC* bufferDesc = CRC::As<CRC_BUFFER_DESC>(&desc);
    if (!bufferDesc) return nullptr;

    std::unique_ptr<CRCID3D11Buffer> buffer = std::make_unique<CRCID3D11Buffer>();

    HRESULT hr = bufferDesc->device_->CreateBuffer
    (
        &bufferDesc->Desc(), &bufferDesc->InitialData(), buffer->d3d11Buffer.GetAddressOf()
    );
    if (FAILED(hr)) return nullptr;

    return buffer;
}

void *CRCID3D11Buffer::GetMem() const
{
    return nullptr;
}

std::size_t CRCID3D11Buffer::GetSize() const
{
    return 0;
}
