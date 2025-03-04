#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_texture.cuh"

std::unique_ptr<ICRCContainable> CRCTexture2DFactoryL0_0::Create(IDESC &desc) const
{
    CRC_TEXTURE2D_DESC* textureDesc = CRC::As<CRC_TEXTURE2D_DESC>(&desc);
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
    byteWidth_ = CRC::GetBytesPerPixel(src.Format) * src.Width * src.Height;

    desc_ = src;
    rcType_ = CRC::GetCRCResourceType(src);

    Malloc(byteWidth_);
    if (desc.initialData_.pSysMem)
    {
        CRC::CheckCuda(cudaMemcpy
        (
            cudaArray_, desc.initialData_.pSysMem, byteWidth_, cudaMemcpyHostToDevice
        ));
    }

#ifndef NDEBUG
    std::string rcTypeStr = CRC::GetCRCResourceTypeString(rcType_);
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

#ifndef NDEBUG
    CRC::Cout("Texture2D destroyed.");
#endif
}

HRESULT CRCTexture2D::GetType(UINT &rcType)
{
    rcType = rcType_;
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

    if (CRC::NeedsWrite(rcType_))
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

CRCCudaResource::CRCCudaResource(D3D11_TEXTURE2D_DESC &desc)
{
    desc_ = desc;
    rcType_ = CRC::GetCRCResourceType(desc);

#ifndef NDEBUG
    std::string rcTypeStr = CRC::GetCRCResourceTypeString(rcType_);
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

#ifndef NDEBUG
    CRC::Cout("Texture2D destroyed.");
#endif
}

HRESULT CRCCudaResource::GetType(UINT &rcType)
{
    rcType = rcType_;
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

    if (CRC::NeedsWrite(rcType_))
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

    rcType_ = 0;
    byteWidth_ = 0;
    cudaArray_ = nullptr;
    surfaceObject_ = 0;
    textureObject_ = 0;
}

std::unique_ptr<ICRCContainable> CRCID3D11Texture2DFactoryL0_0::Create(IDESC &desc) const
{
    CRC_TEXTURE2D_DESC* textureDesc = CRC::As<CRC_TEXTURE2D_DESC>(&desc);
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

    HRESULT hr = textureDesc->d3d11Device_->CreateTexture2D
    (
        &textureDesc->desc_, &textureDesc->initialData_, texture->Get().GetAddressOf()
    );
    if (FAILED(hr))
    {
#ifndef NDEBUG
        CRC::CoutError("Failed to create texture2d from desc. D3D11Device CreateTexture2D failed.");
#endif
        throw std::runtime_error("Failed to create texture2d from desc. D3D11Device CreateTexture2D failed.");
    }

    return texture;
}

CRCID3D11Texture2D::CRCID3D11Texture2D()
{
#ifndef NDEBUG
    CRC::Cout("Texture2D created.");
#endif
}

CRCID3D11Texture2D::~CRCID3D11Texture2D()
{
#ifndef NDEBUG
    CRC::Cout("Texture2D destroyed.");
#endif
}

HRESULT CRCID3D11Texture2D::GetType(UINT &rcType)
{
    rcType = 0;
    return S_OK;
}

const void CRCID3D11Texture2D::GetDesc(D3D11_TEXTURE2D_DESC *dst)
{
    d3d11Texture2D_->GetDesc(dst);
}