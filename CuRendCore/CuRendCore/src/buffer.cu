#include "CuRendCore/include/pch.h"
#include "CuRendCore/include/funcs.cuh"

#include "CuRendCore/include/buffer.cuh"

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
    resType_ = CRC::GetCRCResourceType(src);
    resType_ |= (UINT)CRC_RESOURCE_TYPE::CRC_RESOURCE;

    if (resType_ & (UINT)CRC_RESOURCE_TYPE::GPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::GPU_W)
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

    if (resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_W)
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
    std::string rcTypeStr = CRC::GetCRCResourceTypeString(resType_);
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

HRESULT CRCBuffer::GetResourceType(UINT& rcType)
{
    rcType = resType_;
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

void *const CRCBuffer::GetHostPtr()
{
    if (resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_W)
    {
        return hPtr_;
    }
    else
    {
#ifndef NDEBUG
        CRC::CoutWarning("This buffer is not CPU readable or writable.");
#endif
        return nullptr;
    }
}

HRESULT CRCBuffer::SendHostToDevice()
{
    if (!dPtr_)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Buffer device memory not allocated.");
#endif
        return E_FAIL;
    }

    if (!hPtr_)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Buffer host memory not allocated.");
#endif
        return E_FAIL;
    }

    if (resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_W)
    {
        CRC::CheckCuda(cudaMemcpy(dPtr_, hPtr_, byteWidth_, cudaMemcpyHostToDevice));
    }
    else
    {
#ifndef NDEBUG
        CRC::CoutWarning("This buffer is not CPU readable or writable.");
#endif
        return E_FAIL;
    }

    return S_OK;
}

HRESULT CRCBuffer::SendHostToDevice(const void *src, UINT srcByteWidth)
{
    if (!dPtr_)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Buffer device memory not allocated.");
#endif
        return E_FAIL;
    }

    if (resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_W)
    {
        if (srcByteWidth > byteWidth_)
        {
#ifndef NDEBUG
            CRC::CoutWarning("Source byte width is larger than buffer byte width.");
#endif
            return E_FAIL;
        }

        CRC::CheckCuda(cudaMemcpy(dPtr_, src, srcByteWidth, cudaMemcpyHostToDevice));
    }
    else
    {
#ifndef NDEBUG
        CRC::CoutWarning("This buffer is not CPU readable or writable.");
#endif
        return E_FAIL;
    }

    return S_OK;
}

HRESULT CRCBuffer::SendDeviceToHost()
{
    if (!dPtr_)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Buffer device memory not allocated.");
#endif
        return E_FAIL;
    }

    if (!hPtr_)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Buffer host memory not allocated.");
#endif
        return E_FAIL;
    }

    if (resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_W)
    {
        CRC::CheckCuda(cudaMemcpy(hPtr_, dPtr_, byteWidth_, cudaMemcpyDeviceToHost));
    }
    else
    {
#ifndef NDEBUG
        CRC::CoutWarning("This buffer is not CPU readable or writable.");
#endif
        return E_FAIL;
    }

    return S_OK;
}

bool CRCBuffer::IsCpuAccessible()
{
    return resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_W;
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

    D3D11_BUFFER_DESC d3d11Desc;
    buffer->GetDesc(&d3d11Desc);

    UINT resType = 0;
    resType = CRC::GetCRCResourceType(d3d11Desc);
    resType |= (UINT)CRC_RESOURCE_TYPE::D3D11_RESOURCE;
    buffer->SetResourceType(resType);

#ifndef NDEBUG
    std::string rcTypeStr = CRC::GetCRCResourceTypeString(resType);
    CRC::Cout
    (
        "ID3D11Buffer created.", "\n",
        "Resource Type :", rcTypeStr
    );
#endif

    return buffer;
}

const void CRCID3D11Buffer::GetDesc(D3D11_BUFFER_DESC *dst)
{
    d3d11Buffer_->GetDesc(dst);
}

CRCID3D11Buffer::~CRCID3D11Buffer()
{
#ifndef NDEBUG
    CRC::Cout("ID3D11Buffer destroyed.");
#endif
}

HRESULT CRCID3D11Buffer::GetResourceType(UINT &rcType)
{
    rcType = 0;
    return S_OK;
}

Microsoft::WRL::ComPtr<ID3D11Resource> CRCID3D11Buffer::GetResource()
{
    Microsoft::WRL::ComPtr<ID3D11Resource> resource;
    d3d11Buffer_.As(&resource);
    return resource;
}
