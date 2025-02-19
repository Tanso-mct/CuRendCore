#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_buffer.cuh"

std::unique_ptr<ICRCContainable> CRCBufferFactory::Create(IDESC &desc) const
{
    std::unique_ptr<CRCBuffer> buffer = std::make_unique<CRCBuffer>();
    return buffer;
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

    D3D11_BUFFER_DESC d3d11BufferDesc = {};
    d3d11BufferDesc.ByteWidth = bufferDesc->byteWidth_;
    d3d11BufferDesc.Usage = static_cast<D3D11_USAGE>(bufferDesc->usage_);
    d3d11BufferDesc.BindFlags = bufferDesc->bindFlags_;
    d3d11BufferDesc.CPUAccessFlags = bufferDesc->cpuAccessFlags_;
    d3d11BufferDesc.MiscFlags = bufferDesc->miscFlags_;
    d3d11BufferDesc.StructureByteStride = bufferDesc->structureByteStride_;

    D3D11_SUBRESOURCE_DATA d3d11SubresourceData = {};
    d3d11SubresourceData.pSysMem = bufferDesc->initialData_.pSysMem_;
    d3d11SubresourceData.SysMemPitch = bufferDesc->initialData_.sysMemPitch_;
    d3d11SubresourceData.SysMemSlicePitch = bufferDesc->initialData_.sysMemSlicePitch_;

    HRESULT hr = bufferDesc->device_->CreateBuffer
    (
        &d3d11BufferDesc, &d3d11SubresourceData, buffer->d3d11Buffer.GetAddressOf()
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
