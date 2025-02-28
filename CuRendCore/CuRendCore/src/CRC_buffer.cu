#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_buffer.cuh"

std::unique_ptr<ICRCContainable> CRCBufferFactoryL0_0::Create(IDESC &desc) const
{
    CRC_BUFFER_DESC* bufferDesc = CRC::As<CRC_BUFFER_DESC>(&desc);
    if (!bufferDesc) return nullptr;

    std::unique_ptr<CRCBuffer> buffer = std::make_unique<CRCBuffer>(*bufferDesc);
    return buffer;
}

CRCBuffer::CRCBuffer()
{
}

CRCBuffer::CRCBuffer(CRC_BUFFER_DESC &desc)
{
    Malloc(desc.ByteWidth());
    
    if (desc.SysMem())
    {
        CRC::CheckCuda(cudaMemcpy
        (
            memPtr_, desc.SysMem(), desc.ByteWidth(), cudaMemcpyHostToDevice
        ));
    }

    desc_ = desc.Desc();
}

HRESULT CRCBuffer::GetType(D3D11_RESOURCE_DIMENSION &type)
{
    type = D3D11_RESOURCE_DIMENSION_BUFFER;
    return S_OK;
}

const void CRCBuffer::GetDesc(D3D11_BUFFER_DESC *dst)
{
    std::memcpy(dst, &desc_, sizeof(D3D11_BUFFER_DESC));
}

void CRCBuffer::Malloc(UINT byteWidth, UINT pitch, UINT slicePitch, DXGI_FORMAT format)
{
    if (memPtr_)
    {
#ifndef NDEBUG
        CRC::CoutError("Buffer device memory already allocated.");
#endif
        throw std::runtime_error("Buffer device memory already allocated.");
    }

    byteWidth_ = byteWidth;
    CRC::CheckCuda(cudaMalloc(&memPtr_, byteWidth_));

#ifndef NDEBUG
    CRC::Cout
    (
        "Buffer device memory allocated.", "\n",
        "ByteWidth :", byteWidth_, "\n",
        "Pitch :", pitch, "\n",
        "SlicePitch :", slicePitch
    );
#endif
}

void CRCBuffer::Free()
{
    if (!memPtr_)
    {
#ifndef NDEBUG
        CRC::CoutError("Buffer device memory not allocated.");
#endif
        throw std::runtime_error("Buffer device memory not allocated.");
    }

    byteWidth_ = 0;

    CRC::CheckCuda(cudaFree(memPtr_));
    memPtr_ = nullptr;
}

std::unique_ptr<ICRCContainable> CRCID3D11BufferFactoryL0_0::Create(IDESC &desc) const
{
    CRC_BUFFER_DESC* bufferDesc = CRC::As<CRC_BUFFER_DESC>(&desc);
    if (!bufferDesc) return nullptr;

    std::unique_ptr<CRCID3D11Buffer> buffer = std::make_unique<CRCID3D11Buffer>();

    D3D11_SUBRESOURCE_DATA* initialData = nullptr;
    if (bufferDesc->SysMem()) initialData = &bufferDesc->InitialData();

    if (!bufferDesc->d3d11Device_) throw std::runtime_error("Device not set.");

    HRESULT hr = bufferDesc->d3d11Device_->CreateBuffer
    (
        &bufferDesc->Desc(), initialData, buffer->Get().GetAddressOf()
    );
    if (FAILED(hr)) return nullptr;

    return buffer;
}

Microsoft::WRL::ComPtr<ID3D11Resource> &CRCID3D11Buffer::GetResource()
{
    Microsoft::WRL::ComPtr<ID3D11Resource> resource;
    d3d11Buffer_.As(&resource);

    return resource;
}

HRESULT CRCID3D11Buffer::GetType(D3D11_RESOURCE_DIMENSION & type)
{
    d3d11Buffer_->GetType(&type);
    return S_OK;
}

const void CRCID3D11Buffer::GetDesc(D3D11_BUFFER_DESC *dst)
{
    d3d11Buffer_->GetDesc(dst);
}

const UINT &CRCID3D11Buffer::GetByteWidth() const
{
    D3D11_BUFFER_DESC desc;
    d3d11Buffer_->GetDesc(&desc);

    return desc.ByteWidth;
}