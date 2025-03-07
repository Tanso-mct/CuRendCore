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
    rcType_ = CRC::GetCRCResourceType(src);

    if (rcType_ & (UINT)CRC_RESOURCE_TYPE::BUFFER_GPU_R || rcType_ & (UINT)CRC_RESOURCE_TYPE::BUFFER_GPU_W)
    {
        Malloc(src.ByteWidth);
        if (desc.initialData_.pSysMem)
        {
            CRC::CheckCuda(cudaMemcpy
            (
                dPtr_, desc.initialData_.pSysMem, byteWidth_, cudaMemcpyHostToDevice
            ));
        }
    }

    if (rcType_ & (UINT)CRC_RESOURCE_TYPE::BUFFER_CPU_R || rcType_ & (UINT)CRC_RESOURCE_TYPE::BUFFER_CPU_W)
    {
        HostMalloc(src.ByteWidth);
        if (desc.initialData_.pSysMem)
        {
            CRC::CheckCuda(cudaMemcpy
            (
                hPtr_, desc.initialData_.pSysMem, byteWidth_, cudaMemcpyHostToHost
            ));
        }
    }

#ifndef NDEBUG
    std::string rcTypeStr = CRC::GetCRCResourceTypeString(rcType_);
    CRC::Cout
    (
        "Buffer created.", "\n",
        "Resource Type :", rcTypeStr
    );
#endif
}

CRCBuffer::~CRCBuffer()
{
    if (dPtr_) Free();
    if (hPtr_) HostFree();

#ifndef NDEBUG
    CRC::Cout("Buffer destroyed.");
#endif
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
    if (dPtr_)
    {
#ifndef NDEBUG
        CRC::CoutError("Buffer device memory already allocated.");
#endif
        throw std::runtime_error("Buffer device memory already allocated.");
    }

    byteWidth_ = byteWidth;
    CRC::CheckCuda(cudaMalloc(&dPtr_, byteWidth_));

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
    if (!dPtr_)
    {
#ifndef NDEBUG
        CRC::CoutError("Buffer device memory not allocated.");
#endif
        throw std::runtime_error("Buffer device memory not allocated.");
    }

    byteWidth_ = 0;

    CRC::CheckCuda(cudaFree(dPtr_));
    dPtr_ = nullptr;

#ifndef NDEBUG
    CRC::Cout("Buffer device memory free.");
#endif
}

void CRCBuffer::HostMalloc(UINT byteWidth)
{
    if (hPtr_)
    {
#ifndef NDEBUG
        CRC::CoutError("Buffer host memory already allocated.");
#endif
        throw std::runtime_error("Buffer host memory already allocated.");
    }

    byteWidth_ = byteWidth;
    CRC::CheckCuda(cudaMallocHost(&hPtr_, byteWidth_));

#ifndef NDEBUG
    CRC::Cout
    (
        "Buffer host memory allocated.", "\n",
        "ByteWidth :", byteWidth_
    );
#endif
}

void CRCBuffer::HostFree()
{
    if (!hPtr_)
    {
#ifndef NDEBUG
        CRC::CoutError("Buffer host memory not allocated.");
#endif
        throw std::runtime_error("Buffer host memory not allocated.");
    }

    byteWidth_ = 0;

    CRC::CheckCuda(cudaFreeHost(hPtr_));
    hPtr_ = nullptr;
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

    HRESULT hr;
    if (bufferDesc->initialData_.pSysMem)
    {
        hr = bufferDesc->d3d11Device_->CreateBuffer
        (
            &bufferDesc->desc_, &bufferDesc->initialData_, &buffer->Get()
        );
    }
    else
    {
        hr = bufferDesc->d3d11Device_->CreateBuffer
        (
            &bufferDesc->desc_, nullptr, &buffer->Get()
        );
    }

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

CRCID3D11Buffer::CRCID3D11Buffer()
{
#ifndef NDEBUG
    CRC::Cout("ID3D11Buffer created.");
#endif
}

CRCID3D11Buffer::~CRCID3D11Buffer()
{
#ifndef NDEBUG
    CRC::Cout("ID3D11Buffer destroyed.");
#endif
}

HRESULT CRCID3D11Buffer::GetType(UINT &rcType)
{
    rcType = 0;
    return S_OK;
}