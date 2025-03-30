#include "CuRendCore/include/pch.h"
#include "CuRendCore/include/funcs.cuh"

#include "CuRendCore/include/texture.cuh"

std::unique_ptr<ICRCContainable> CRCTexture2DFactoryL0_0::Create(IDESC &desc) const
{
    CRC_TEXTURE2D_DESC* textureDesc = WACore::As<CRC_TEXTURE2D_DESC>(&desc);
    if (!textureDesc)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to create texture2d from desc. Desc is not CRC_TEXTURE2D_DESC.");
#endif
        return nullptr;
    }

    std::unique_ptr<CRCTexture2D> texture = std::make_unique<CRCTexture2D>(*textureDesc);
    return texture;
}

CRCTexture2D::CRCTexture2D(CRC_TEXTURE2D_DESC& desc)
{
    D3D11_TEXTURE2D_DESC& src = desc.desc_;

    desc_ = src;
    resType_ = CRC::GetCRCResourceType(src);
    resType_ |= (UINT)CRC_RESOURCE_TYPE::CRC_RESOURCE;

    if (resType_ & (UINT)CRC_RESOURCE_TYPE::GPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::GPU_W)
    {
        Malloc(CRC::GetBytesPerPixel(src.Format) * src.Width * src.Height);
        if (desc.initialData_.pSysMem)
        {
            CRC::CheckCuda(cudaMemcpy2DToArray
            (
                cudaArray_, 0, 0, desc.initialData_.pSysMem, desc_.Width * CRC::GetBytesPerPixel(desc_.Format),
                desc_.Width * CRC::GetBytesPerPixel(desc_.Format), desc_.Height, cudaMemcpyHostToDevice
            ));
        }
    }

    if (resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_W)
    {
        HostMalloc(CRC::GetBytesPerPixel(src.Format) * src.Width * src.Height);
        if (desc.initialData_.pSysMem)
        {
            CRC::CheckCuda(cudaMemcpy2DFromArray
            (
                hPtr_, desc_.Width * CRC::GetBytesPerPixel(desc_.Format), cudaArray_, 0, 0,
                desc_.Width * CRC::GetBytesPerPixel(desc_.Format), desc_.Height, cudaMemcpyDeviceToHost
            ));
        }
    }

#ifndef NDEBUG
    std::string rcTypeStr = CRC::GetCRCResourceTypeString(resType_);
    CRC::Cout
    (
        "Texture2D created.", "\n",
        "Resource Type :", rcTypeStr
    );
#endif
}

CRCTexture2D::~CRCTexture2D()
{
    if (cudaArray_) Free();
    if (hPtr_) HostFree();

#ifndef NDEBUG
    CRC::Cout("Texture2D destroyed.");
#endif
}

HRESULT CRCTexture2D::GetResourceType(UINT &rcType)
{
    rcType = resType_;
    return S_OK;
}

const void CRCTexture2D::GetDesc(D3D11_TEXTURE2D_DESC *dst)
{
    std::memcpy(dst, &desc_, sizeof(D3D11_TEXTURE2D_DESC));
}

void CRCTexture2D::Malloc(UINT byteWidth)
{
    if (cudaArray_)
    {
#ifndef NDEBUG
        CRC::CoutError("Texture2D device memory already allocated.");
#endif
        throw std::runtime_error("Texture2D device memory already allocated.");
    }

    byteWidth_ = byteWidth;

    cudaChannelFormatDesc channelDesc;
    CRC::CreateCudaChannelDescFromDXGIFormat(channelDesc, desc_.Format);

    CRC::CheckCuda(cudaMallocArray(&cudaArray_, &channelDesc, desc_.Width, desc_.Height));

    struct cudaResourceDesc resDesc;
    ZeroMemory(&resDesc, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cudaArray_;

    if (CRC::NeedsWrite(resType_))
    {
        CRC::CheckCuda(cudaCreateSurfaceObject(&surfaceObject_, &resDesc));
    }
    else
    {
        struct cudaTextureDesc texDesc;
        ZeroMemory(&texDesc, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;

        CRC::CheckCuda(cudaCreateTextureObject(&textureObject_, &resDesc, &texDesc, nullptr));
    }

#ifndef NDEBUG
    CRC::Cout
    (
        "Texture2D device memory allocated.", "\n",
        "ByteWidth :", byteWidth_, "\n",
        "Width :", desc_.Width, "\n",
        "Height :", desc_.Height
    );
#endif
}

void CRCTexture2D::Free()
{
    if (!cudaArray_)
    {
#ifndef NDEBUG
        CRC::CoutError("Texture2D device memory not allocated.");
#endif
        throw std::runtime_error("Texture2D device memory not allocated.");
    }

    byteWidth_ = 0;
    surfaceObject_ = 0;
    textureObject_ = 0;

    CRC::CheckCuda(cudaFreeArray(cudaArray_));
    cudaArray_ = nullptr;

#ifndef NDEBUG
    CRC::Cout("Texture2D device memory free.");
#endif
}

void CRCTexture2D::HostMalloc(UINT byteWidth)
{
    if (hPtr_)
    {
#ifndef NDEBUG
        CRC::CoutError("Texture2D host memory already allocated.");
#endif
        throw std::runtime_error("Texture2D host memory already allocated.");
    }

    byteWidth_ = byteWidth;
    CRC::CheckCuda(cudaMallocHost(&hPtr_, byteWidth_));

#ifndef NDEBUG
    CRC::Cout
    (
        "Texture2D host memory allocated.", "\n",
        "ByteWidth :", byteWidth_, "\n",
        "Width :", desc_.Width, "\n",
        "Height :", desc_.Height
    );
#endif
}

void CRCTexture2D::HostFree()
{
    if (!hPtr_)
    {
#ifndef NDEBUG
        CRC::CoutError("Texture2D host memory not allocated.");
#endif
        throw std::runtime_error("Texture2D host memory not allocated.");
    }

    byteWidth_ = 0;

    CRC::CheckCuda(cudaFreeHost(hPtr_));
    hPtr_ = nullptr;

#ifndef NDEBUG
    CRC::Cout("Texture2D host memory free.");
#endif
}

const UINT &CRCTexture2D::GetRowPitch()
{
    return desc_.Width * CRC::GetBytesPerPixel(desc_.Format);
}

void *const CRCTexture2D::GetHostPtr()
{
    if (resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_W)
    {
        return hPtr_;
    }
    else
    {
#ifndef NDEBUG
        CRC::CoutWarning("This texture2d is not CPU readable or writable.");
#endif
        return nullptr;
    }
}

HRESULT CRCTexture2D::SendHostToDevice()
{
    if (!cudaArray_)
    {
#ifndef NDEBUG
        CRC::CoutError("Texture2D device memory not allocated.");
#endif
        return E_FAIL;
    }

    if (!hPtr_)
    {
#ifndef NDEBUG
        CRC::CoutError("Texture2D host memory not allocated.");
#endif
        return E_FAIL;
    }

    if (resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_W)
    {
        CRC::CheckCuda(cudaMemcpy2DToArray
        (
            cudaArray_, 0, 0, hPtr_, desc_.Width * CRC::GetBytesPerPixel(desc_.Format),
            desc_.Width * CRC::GetBytesPerPixel(desc_.Format), desc_.Height, cudaMemcpyHostToDevice
        ));
    }
    else
    {
#ifndef NDEBUG
        CRC::CoutWarning("This texture2d is not CPU readable or writable.");
#endif
        return E_FAIL;
    }

    return S_OK;
}

HRESULT CRCTexture2D::SendHostToDevice(const void *src, UINT srcByteWidth)
{
    if (!cudaArray_)
    {
#ifndef NDEBUG
        CRC::CoutError("Texture2D device memory not allocated.");
#endif
        return E_FAIL;
    }

    if (resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_W)
    {
        if (srcByteWidth > byteWidth_)
        {
#ifndef NDEBUG
            CRC::CoutError("Failed to send host to device. Source byte width is larger than destination.");
#endif
            return E_FAIL;
        }

        CRC::CheckCuda(cudaMemcpy2DToArray
        (
            cudaArray_, 0, 0, src, desc_.Width * CRC::GetBytesPerPixel(desc_.Format),
            desc_.Width * CRC::GetBytesPerPixel(desc_.Format), desc_.Height, cudaMemcpyHostToDevice
        ));
    }
    else
    {
#ifndef NDEBUG
        CRC::CoutWarning("This texture2d is not CPU readable or writable.");
#endif
        return E_FAIL;
    }

    return S_OK;
}

HRESULT CRCTexture2D::SendDeviceToHost()
{
    if (!cudaArray_)
    {
#ifndef NDEBUG
        CRC::CoutError("Texture2D device memory not allocated.");
#endif
        return E_FAIL;
    }

    if (!hPtr_)
    {
#ifndef NDEBUG
        CRC::CoutError("Texture2D host memory not allocated.");
#endif
        return E_FAIL;
    }

    if (resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_W)
    {
        CRC::CheckCuda(cudaMemcpy2DFromArray
        (
            hPtr_, desc_.Width * CRC::GetBytesPerPixel(desc_.Format), cudaArray_, 0, 0,
            desc_.Width * CRC::GetBytesPerPixel(desc_.Format), desc_.Height, cudaMemcpyDeviceToHost
        ));
    }
    else
    {
#ifndef NDEBUG
        CRC::CoutWarning("This texture2d is not CPU readable or writable.");
#endif
        return E_FAIL;
    }

    return S_OK;
}

bool CRCTexture2D::IsCpuAccessible()
{
    return resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_W;
}

CRCCudaResource::CRCCudaResource(D3D11_TEXTURE2D_DESC &desc)
{
    desc_ = desc;
    resType_ = CRC::GetCRCResourceType(desc);
    resType_ |= (UINT)CRC_RESOURCE_TYPE::CRC_RESOURCE;

    if (resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_W)
    {
        HostMalloc(CRC::GetBytesPerPixel(desc_.Format) * desc_.Width * desc_.Height);
    }

#ifndef NDEBUG
    std::string rcTypeStr = CRC::GetCRCResourceTypeString(resType_);
    CRC::Cout
    (
        "Texture2D created.", "\n",
        "Resource Type :", rcTypeStr
    );
#endif
}

CRCCudaResource::~CRCCudaResource()
{
    if (cudaArray_) Unassign();
    if (hPtr_) HostFree();

#ifndef NDEBUG
    CRC::Cout("Texture2D destroyed.");
#endif
}

HRESULT CRCCudaResource::GetResourceType(UINT &rcType)
{
    rcType = resType_;
    return S_OK;
}

const void CRCCudaResource::GetDesc(D3D11_TEXTURE2D_DESC *dst)
{
    std::memcpy(dst, &desc_, sizeof(D3D11_TEXTURE2D_DESC));
}

void CRCCudaResource::Assign(void *const mem, UINT byteWidth)
{
    if (cudaArray_)
    {
#ifndef NDEBUG
        CRC::CoutError("Texture2D device memory already assigned.");
#endif
        throw std::runtime_error("Texture2D device memory already assigned.");
    }

    byteWidth_ = byteWidth;

    cudaArray_ = reinterpret_cast<cudaArray*>(mem);
    if (!cudaArray_)
    {
#ifndef NDEBUG
        CRC::CoutError("Failed to cast cudaArray.");
#endif
        throw std::runtime_error("Failed to cast cudaArray.");
    }

    struct cudaResourceDesc resDesc;
    ZeroMemory(&resDesc, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cudaArray_;

    if (CRC::NeedsWrite(resType_))
    {
        CRC::CheckCuda(cudaCreateSurfaceObject(&surfaceObject_, &resDesc));
    }
    else
    {
        struct cudaTextureDesc texDesc;
        ZeroMemory(&texDesc, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;

        CRC::CheckCuda(cudaCreateTextureObject(&textureObject_, &resDesc, &texDesc, nullptr));
    }
}

void CRCCudaResource::Unassign()
{
    if (!cudaArray_)
    {
#ifndef NDEBUG
        CRC::CoutError("Texture2D device memory not assigned.");
#endif
        throw std::runtime_error("Texture2D device memory not assigned.");
    }

    resType_ = 0;
    byteWidth_ = 0;
    cudaArray_ = nullptr;
    surfaceObject_ = 0;
    textureObject_ = 0;
}

void CRCCudaResource::HostMalloc(UINT byteWidth)
{
    if (hPtr_)
    {
#ifndef NDEBUG
        CRC::CoutError("CudaResource host memory already allocated.");
#endif
        throw std::runtime_error("CudaResource host memory already allocated.");
    }

    byteWidth_ = byteWidth;
    CRC::CheckCuda(cudaMallocHost(&hPtr_, byteWidth_));

#ifndef NDEBUG
    CRC::Cout
    (
        "CudaResource host memory allocated.", "\n",
        "ByteWidth :", byteWidth_, "\n",
        "Width :", desc_.Width, "\n",
        "Height :", desc_.Height
    );
#endif
}

void CRCCudaResource::HostFree()
{
    if (!hPtr_)
    {
#ifndef NDEBUG
        CRC::CoutError("CudaResource host memory not allocated.");
#endif
        throw std::runtime_error("CudaResource host memory not allocated.");
    }

    byteWidth_ = 0;

    CRC::CheckCuda(cudaFreeHost(hPtr_));
    hPtr_ = nullptr;

#ifndef NDEBUG
    CRC::Cout("CudaResource host memory free.");
#endif
}

const UINT &CRCCudaResource::GetRowPitch()
{
    return desc_.Width * CRC::GetBytesPerPixel(desc_.Format);
}

void *const CRCCudaResource::GetHostPtr()
{
    if (resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_W)
    {
        return hPtr_;
    }
    else
    {
#ifndef NDEBUG
        CRC::CoutWarning("This CudaResource is not CPU readable or writable.");
#endif
        return nullptr;
    }
}

HRESULT CRCCudaResource::SendHostToDevice()
{
    if (!cudaArray_)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Texture2D device memory not allocated.");
#endif
        return E_FAIL;
    }

    if (!hPtr_)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Texture2D host memory not allocated.");
#endif
        return E_FAIL;
    }

    if (resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_W)
    {
        CRC::CheckCuda(cudaMemcpy2DToArray
        (
            cudaArray_, 0, 0, hPtr_, desc_.Width * CRC::GetBytesPerPixel(desc_.Format),
            desc_.Width * CRC::GetBytesPerPixel(desc_.Format), desc_.Height, cudaMemcpyHostToDevice
        ));
    }
    else
    {
#ifndef NDEBUG
        CRC::CoutWarning("This CudaResource is not CPU readable or writable.");
#endif
        return E_FAIL;
    }

    return S_OK;
}

HRESULT CRCCudaResource::SendHostToDevice(const void *src, UINT srcByteWidth)
{
    if (!cudaArray_)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Texture2D device memory not allocated.");
#endif
        return E_FAIL;
    }

    if (resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_W)
    {
        if (srcByteWidth > byteWidth_)
        {
#ifndef NDEBUG
            CRC::CoutWarning("Failed to send host to device. Source byte width is larger than destination.");
#endif
            return E_FAIL;
        }

        CRC::CheckCuda(cudaMemcpy2DToArray
        (
            cudaArray_, 0, 0, src, desc_.Width * CRC::GetBytesPerPixel(desc_.Format),
            desc_.Width * CRC::GetBytesPerPixel(desc_.Format), desc_.Height, cudaMemcpyHostToDevice
        ));
    }
    else
    {
#ifndef NDEBUG
        CRC::CoutWarning("This CudaResource is not CPU readable or writable.");
#endif
        return E_FAIL;
    }

    return S_OK;
}

HRESULT CRCCudaResource::SendDeviceToHost()
{
    if (!cudaArray_)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Texture2D device memory not allocated.");
#endif
        return E_FAIL;
    }

    if (!hPtr_)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Texture2D host memory not allocated.");
#endif
        return E_FAIL;
    }

    if (resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_W)
    {
        CRC::CheckCuda(cudaMemcpy2DFromArray
        (
            hPtr_, desc_.Width * CRC::GetBytesPerPixel(desc_.Format), cudaArray_, 0, 0,
            desc_.Width * CRC::GetBytesPerPixel(desc_.Format), desc_.Height, cudaMemcpyDeviceToHost
        ));
    }
    else
    {
#ifndef NDEBUG
        CRC::CoutWarning("This CudaResource is not CPU readable or writable.");
#endif
        return E_FAIL;
    }

    return S_OK;
}

bool CRCCudaResource::IsCpuAccessible()
{
    return resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_R || resType_ & (UINT)CRC_RESOURCE_TYPE::CPU_W;
}

std::unique_ptr<ICRCContainable> CRCID3D11Texture2DFactoryL0_0::Create(IDESC &desc) const
{
    CRC_TEXTURE2D_DESC* textureDesc = WACore::As<CRC_TEXTURE2D_DESC>(&desc);
    if (!textureDesc)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to create texture2d from desc. Desc is not CRC_TEXTURE2D_DESC.");
#endif
        return nullptr;
    }

    if (!textureDesc->d3d11Device_)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to create texture2d from desc. D3D11 device is nullptr.");
#endif
        return nullptr;
    }

    std::unique_ptr<CRCID3D11Texture2D> texture = std::make_unique<CRCID3D11Texture2D>();

    HRESULT hr;
    if (textureDesc->initialData_.pSysMem)
    {
        hr = textureDesc->d3d11Device_->CreateTexture2D
        (
            &textureDesc->desc_, &textureDesc->initialData_, &texture->Get()
        );
    }
    else
    {
        hr = textureDesc->d3d11Device_->CreateTexture2D
        (
            &textureDesc->desc_, nullptr, &texture->Get()
        );
    }

    if (FAILED(hr))
    {
#ifndef NDEBUG
        CRC::CoutError("Failed to create texture2d from desc. D3D11Device CreateTexture2D failed.");
#endif
        throw std::runtime_error("Failed to create texture2d from desc. D3D11Device CreateTexture2D failed.");
    }

    D3D11_TEXTURE2D_DESC d3d11Desc;
    texture->GetDesc(&d3d11Desc);

    UINT resType = 0;
    resType = CRC::GetCRCResourceType(d3d11Desc);
    resType |= (UINT)CRC_RESOURCE_TYPE::D3D11_RESOURCE;
    texture->SetResourceType(resType);

#ifndef NDEBUG
    std::string rcTypeStr = CRC::GetCRCResourceTypeString(resType);
    CRC::Cout
    (
        "ID3D11Texture2D created.", "\n",
        "Resource Type :", rcTypeStr
    );
#endif

    return texture;
}

CRCID3D11Texture2D::~CRCID3D11Texture2D()
{
#ifndef NDEBUG
    CRC::Cout("ID3D11Texture2D destroyed.");
#endif
}

HRESULT CRCID3D11Texture2D::GetResourceType(UINT &resType)
{
    resType = resType_;
    return S_OK;
}

Microsoft::WRL::ComPtr<ID3D11Resource> CRCID3D11Texture2D::GetResource()
{
    Microsoft::WRL::ComPtr<ID3D11Resource> resource;
    d3d11Texture2D_.As(&resource);
    return resource;
}

const void CRCID3D11Texture2D::GetDesc(D3D11_TEXTURE2D_DESC *dst)
{
    d3d11Texture2D_->GetDesc(dst);
}