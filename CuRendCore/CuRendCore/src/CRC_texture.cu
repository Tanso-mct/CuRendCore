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
    Malloc
    (
        CRC::GetBytesPerPixel(desc.Format()) * desc.Width() * desc.Height(), 
        CRC::GetBytesPerPixel(desc.Format()) * desc.Width(), 
        1,
        desc.Format()
    );

    if (desc.SysMem())
    {
        CRC::CheckCuda(cudaMemcpy
        (
            cudaArray_, desc.SysMem(), GetByteWidth(), cudaMemcpyHostToDevice
        ));
    }

    desc_ = desc.Desc();
}

CRCTexture2D::~CRCTexture2D()
{
    if (cudaArray_) Free();
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

    std::unique_ptr<CRCID3D11Texture2D> texture = std::make_unique<CRCID3D11Texture2D>();

    D3D11_SUBRESOURCE_DATA* initialData = nullptr;
    if (textureDesc->SysMem()) initialData = &textureDesc->InitialData();

    if (!textureDesc->d3d11Device_)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to create texture2d from desc. D3D11 device is nullptr.");
#endif
        return nullptr;
    }

    HRESULT hr = textureDesc->d3d11Device_->CreateTexture2D
    (
        &textureDesc->Desc(), initialData, texture->Get().GetAddressOf()
    );
    if (FAILED(hr))
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to create texture2d from desc. D3D11Device CreateTexture2D failed.");
#endif
        return nullptr;
    }

    return texture;
}

Microsoft::WRL::ComPtr<ID3D11Resource> &CRCID3D11Texture2D::GetResource()
{
    Microsoft::WRL::ComPtr<ID3D11Resource> resource;
    d3d11Texture2D_.As(&resource);

    return resource;
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

const UINT &CRCID3D11Texture2D::GetByteWidth() const
{
    D3D11_TEXTURE2D_DESC desc;
    d3d11Texture2D_->GetDesc(&desc);

    return desc.Width;
}

const UINT &CRCID3D11Texture2D::GetPitch() const
{
    D3D11_TEXTURE2D_DESC desc;
    d3d11Texture2D_->GetDesc(&desc);

    return desc.Width * CRC::GetBytesPerPixel(desc.Format);
}

const UINT &CRCID3D11Texture2D::GetSlicePitch() const
{
    D3D11_TEXTURE2D_DESC desc;
    d3d11Texture2D_->GetDesc(&desc);

    return desc.Width * desc.Height * CRC::GetBytesPerPixel(desc.Format);
}

void CRCTexture2D::Malloc(UINT byteWidth, UINT pitch, UINT slicePitch, DXGI_FORMAT format)
{
    if (cudaArray_)
    {
#ifndef NDEBUG
        CRC::CoutError("Texture2D device memory already allocated.");
#endif
        throw std::runtime_error("Texture2D device memory already allocated.");
    }

    byteWidth_ = byteWidth;
    pitch_ = pitch;
    slicePitch_ = slicePitch;

    UINT width = pitch / CRC::GetBytesPerPixel(format);
    UINT height = byteWidth / pitch_;

    cudaChannelFormatDesc channelDesc;
    CRC::CreateCudaChannelDescFromDXGIFormat(channelDesc, format);

    CRC::CheckCuda(cudaMallocArray(&cudaArray_, &channelDesc, width, height));
    
#ifndef NDEBUG
    CRC::Cout
    (
        "Texture2D device memory allocated.", "\n", 
        "ByteWidth :", byteWidth_, "\n",
        "Pitch :", pitch_, "\n",
        "SlicePitch :", slicePitch_
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
    pitch_ = 0;
    slicePitch_ = 0;

    CRC::CheckCuda(cudaFreeArray(cudaArray_));
    cudaArray_ = nullptr;

#ifndef NDEBUG
    CRC::Cout("Texture2D device memory free.");
#endif
}

CRCTexutre2DAttached::~CRCTexutre2DAttached()
{
    if (cudaArray_) Unassign();
}

HRESULT CRCTexutre2DAttached::GetType(UINT &rcType)
{
    rcType = rcType_;
    return S_OK;
}

const void CRCTexutre2DAttached::GetDesc(D3D11_TEXTURE2D_DESC *dst)
{
    std::memcpy(dst, &desc_, sizeof(D3D11_TEXTURE2D_DESC));
}

void CRCTexutre2DAttached::Assign(void *const mem, UINT byteWidth, UINT pitch, UINT slicePitch)
{
    if (cudaArray_)
    {
#ifndef NDEBUG
        CRC::CoutError("Texture2D device memory already allocated.");
#endif
        throw std::runtime_error("Texture2D device memory already allocated.");
    }

    byteWidth_ = byteWidth;
    pitch_ = pitch;
    slicePitch_ = slicePitch;

    cudaArray_ = reinterpret_cast<cudaArray*>(mem);
    if (!cudaArray_)
    {
#ifndef NDEBUG
        CRC::CoutError("Failed to cast cudaArray.");
#endif
        throw std::runtime_error("Failed to cast cudaArray.");
    }

#ifndef NDEBUG
    CRC::Cout
    (
        "Texture2D device memory assigned.", "\n",
        "ByteWidth :", byteWidth_, "\n",
        "Pitch :", pitch_, "\n",
        "SlicePitch :", slicePitch_
    );
#endif
}

void CRCTexutre2DAttached::Unassign()
{
    if (!cudaArray_)
    {
#ifndef NDEBUG
        CRC::CoutError("Texture2D device memory not allocated.");
#endif
        throw std::runtime_error("Texture2D device memory not allocated.");
    }

    byteWidth_ = 0;
    pitch_ = 0;
    slicePitch_ = 0;

    cudaArray_ = nullptr;

#ifndef NDEBUG
    CRC::Cout("Texture2D device memory unassigned.");
#endif
}

CRCSurface2D::~CRCSurface2D()
{
    if (cudaSurface_ != 0) Unassign();
}

HRESULT CRCSurface2D::GetType(UINT &rcType)
{
    rcType = rcType_;
    return S_OK;
}

const void CRCSurface2D::GetDesc(D3D11_TEXTURE2D_DESC *dst)
{
    std::memcpy(dst, &desc_, sizeof(D3D11_TEXTURE2D_DESC));
}

void CRCSurface2D::Assign(void *const mem, UINT byteWidth, UINT pitch, UINT slicePitch)
{
    if (cudaSurface_ != 0)
    {
#ifndef NDEBUG
        CRC::CoutError("Surface2D already attached.");
#endif
        throw std::runtime_error("Surface2D already attached.");
    }

    byteWidth_ = byteWidth;
    pitch_ = pitch;
    slicePitch_ = slicePitch;

    cudaArray_t cudaAry = reinterpret_cast<cudaArray_t>(mem);
    if (!cudaAry)
    {
#ifndef NDEBUG
        CRC::CoutError("Failed to cast cudaArray.");
#endif
        throw std::runtime_error("Failed to cast cudaArray.");
    }

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cudaAry;

    CRC::CheckCuda(cudaCreateSurfaceObject(&cudaSurface_, &resDesc));

#ifndef NDEBUG
    CRC::Cout
    (
        "Surface2D attached.", "\n",
        "ByteWidth :", byteWidth_, "\n",
        "Pitch :", pitch_, "\n",
        "SlicePitch :", slicePitch_
    );
#endif
}

void CRCSurface2D::Unassign()
{
    if (cudaSurface_ == 0)
    {
#ifndef NDEBUG
        CRC::CoutError("Surface2D not attached.");
#endif
        throw std::runtime_error("Surface2D not attached.");
    }

    byteWidth_ = 0;
    pitch_ = 0;
    slicePitch_ = 0;

    cudaSurface_ = 0;

#ifndef NDEBUG
    CRC::Cout("Surface2D unassigned.");
#endif
}
