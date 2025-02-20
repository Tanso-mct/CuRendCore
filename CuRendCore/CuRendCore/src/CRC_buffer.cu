#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_buffer.cuh"

CRC_API void CRC::SetAccess
(
    D3D11_USAGE usage, UINT cpuAccessFlags, 
    CRCAccess& gpuRead, CRCAccess& gpuWrite, CRCAccess& cpuRead, CRCAccess& cpuWrite
){
    if (usage == D3D11_USAGE_DEFAULT)
    {
        gpuRead = CRC::As<CRCAccess>(&CRCAccessEnabled());
        gpuWrite = CRC::As<CRCAccess>(&CRCAccessEnabled());
        cpuRead = std::make_unique<CRCMemAccessDisabled>();
        cpuWrite = std::make_unique<CRCMemAccessDisabled>();
    }
    else if (usage == D3D11_USAGE_IMMUTABLE)
    {
        gpuRead = CRC::As<CRCAccess>(&CRCAccessEnabled());
        gpuWrite = std::make_unique<CRCMemAccessDisabled>();
        cpuRead = std::make_unique<CRCMemAccessDisabled>();
        cpuWrite = std::make_unique<CRCMemAccessDisabled>();
    }
    else if (usage == D3D11_USAGE_DYNAMIC)
    {
        gpuRead = CRC::As<CRCAccess>(&CRCAccessEnabled());
        gpuWrite = std::make_unique<CRCMemAccessDisabled>();
        cpuRead = std::make_unique<CRCMemAccessDisabled>();
        cpuWrite = CRC::As<CRCAccess>(&CRCAccessEnabled());
    }
    else if (usage == D3D11_USAGE_STAGING)
    {
        gpuRead = CRC::As<CRCAccess>(&CRCAccessEnabled());
        gpuWrite = CRC::As<CRCAccess>(&CRCAccessEnabled());
        cpuRead = CRC::As<CRCAccess>(&CRCAccessEnabled());
        cpuWrite = CRC::As<CRCAccess>(&CRCAccessEnabled());
    }


}

std::unique_ptr<ICRCContainable> CRCBufferFactory::Create(IDESC &desc) const
{
    CRC_BUFFER_DESC* bufferDesc = CRC::As<CRC_BUFFER_DESC>(&desc);
    if (!bufferDesc) return nullptr;

    std::unique_ptr<CRCBuffer> buffer = std::make_unique<CRCBuffer>();

    CRC::MallocMem(buffer->mem, bufferDesc->Desc().ByteWidth);
    CRC::SetDeviceMem(buffer->mem, bufferDesc->InitialData());

    return buffer;
}

CRCBuffer::~CRCBuffer()
{
    CRC::FreeDeviceMem(mem);
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
