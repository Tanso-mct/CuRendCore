#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_buffer.cuh"

std::unique_ptr<ICRCContainable> CRCBufferFactory::Create(IDESC &desc) const
{
    CRC_BUFFER_DESC* bufferDesc = CRC::As<CRC_BUFFER_DESC>(&desc);
    if (!bufferDesc) return nullptr;

    std::unique_ptr<CRCBuffer> buffer = std::make_unique<CRCBuffer>();
    buffer->dMem = std::make_unique<CRCDeviceMem>();

    buffer->dMem->Malloc(bufferDesc->ByteWidth(), 1, 1);
    
    if (bufferDesc->SysMem())
    {
        CRC::CheckCuda(cudaMemcpy
        (
            buffer->dMem.get(), bufferDesc->SysMem(), bufferDesc->ByteWidth(), cudaMemcpyHostToDevice
        ));
    }

    buffer->desc_ = bufferDesc->Desc();

    return buffer;
}

const void CRCBuffer::GetDesc(D3D11_BUFFER_DESC *dst)
{
    std::memcpy(dst, &desc_, sizeof(D3D11_BUFFER_DESC));
}


std::unique_ptr<ICRCContainable> CRCID3D11BufferFactory::Create(IDESC &desc) const
{
    CRC_BUFFER_DESC* bufferDesc = CRC::As<CRC_BUFFER_DESC>(&desc);
    if (!bufferDesc) return nullptr;

    std::unique_ptr<CRCID3D11Buffer> buffer = std::make_unique<CRCID3D11Buffer>();

    D3D11_SUBRESOURCE_DATA* initialData = nullptr;
    if (bufferDesc->SysMem()) initialData = &bufferDesc->InitialData();

    HRESULT hr = bufferDesc->device_->CreateBuffer
    (
        &bufferDesc->Desc(), initialData, buffer->d3d11Buffer.GetAddressOf()
    );
    if (FAILED(hr)) return nullptr;

    return buffer;
}

Microsoft::WRL::ComPtr<ID3D11Resource> &CRCID3D11Buffer::GetResource()
{
    Microsoft::WRL::ComPtr<ID3D11Resource> resource;
    d3d11Buffer.As(&resource);

    return resource;
}

const UINT &CRCID3D11Buffer::GetByteWidth() const
{
    D3D11_BUFFER_DESC desc;
    d3d11Buffer->GetDesc(&desc);

    return desc.ByteWidth;
}

const void CRCID3D11Buffer::GetDesc(D3D11_BUFFER_DESC *dst)
{
    d3d11Buffer->GetDesc(dst);
}