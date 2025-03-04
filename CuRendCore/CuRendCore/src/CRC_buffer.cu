#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_buffer.cuh"

std::unique_ptr<ICRCContainable> CRCBufferFactoryL0_0::Create(IDESC &desc) const
{
    CRC_BUFFER_DESC* bufferDesc = CRC::As<CRC_BUFFER_DESC>(&desc);
    if (!bufferDesc)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to create buffer from desc. Desc is not CRC_BUFFER_DESC.");
#endif
        return nullptr;
    }

    std::unique_ptr<CRCBuffer> buffer = std::make_unique<CRCBuffer>(*bufferDesc);
    return buffer;
}

CRCBuffer::CRCBuffer(CRC_BUFFER_DESC &desc)
{
    D3D11_BUFFER_DESC& src = desc.desc_;
    desc_ = src;

    Malloc(src.ByteWidth);
    if (desc.initialData_.pSysMem)
    {
        CRC::CheckCuda(cudaMemcpy
        (
            memPtr_, desc.initialData_.pSysMem, byteWidth_, cudaMemcpyHostToDevice
        ));
    }
}

CRCBuffer::~CRCBuffer()
{
    if (memPtr_) Free();
}

HRESULT CRCBuffer::GetType(UINT& rcType)
{
    rcType = rcType_;
    return S_OK;
}

const void CRCBuffer::GetDesc(D3D11_BUFFER_DESC *dst)
{
    std::memcpy(dst, &desc_, sizeof(D3D11_BUFFER_DESC));
}

void CRCBuffer::Malloc(UINT byteWidth)
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
        "ByteWidth :", byteWidth_
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

#ifndef NDEBUG
    CRC::Cout("Buffer device memory free.");
#endif
}

std::unique_ptr<ICRCContainable> CRCID3D11BufferFactoryL0_0::Create(IDESC &desc) const
{
    CRC_BUFFER_DESC* bufferDesc = CRC::As<CRC_BUFFER_DESC>(&desc);
    if (!bufferDesc)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to create buffer from desc. Desc is not CRC_BUFFER_DESC.");
#endif
        return nullptr;
    }

    if (!bufferDesc->d3d11Device_)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to create buffer. D3D11 device is nullptr.");
#endif
        return nullptr;
    }

    std::unique_ptr<CRCID3D11Buffer> buffer = std::make_unique<CRCID3D11Buffer>();

    HRESULT hr = bufferDesc->d3d11Device_->CreateBuffer
    (
        &bufferDesc->desc_, &bufferDesc->initialData_, buffer->Get().GetAddressOf()
    );
    if (FAILED(hr))
    {
#ifndef NDEBUG
        CRC::CoutError("Failed to create buffer.");
#endif
        throw std::runtime_error("Failed to create buffer.");
    }

    return buffer;
}

const void CRCID3D11Buffer::GetDesc(D3D11_BUFFER_DESC *dst)
{
    d3d11Buffer_->GetDesc(dst);
}

HRESULT CRCID3D11Buffer::GetType(UINT& rcType)
{
    rcType = 0;
    return S_OK;
}