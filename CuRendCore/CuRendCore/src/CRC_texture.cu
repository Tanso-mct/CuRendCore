#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_texture.cuh"

std::unique_ptr<ICRCContainable> CRCTexture2DFactoryL0_0::Create(IDESC &desc) const
{
    CRC_TEXTURE2D_DESC* textureDesc = CRC::As<CRC_TEXTURE2D_DESC>(&desc);
    if (!textureDesc) return nullptr;

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

HRESULT CRCTexture2D::GetType(D3D11_RESOURCE_DIMENSION &type)
{
    type = D3D11_RESOURCE_DIMENSION::D3D11_RESOURCE_DIMENSION_TEXTURE2D;
    return S_OK;
}

const void CRCTexture2D::GetDesc(D3D11_TEXTURE2D_DESC *dst)
{
    std::memcpy(dst, &desc_, sizeof(D3D11_TEXTURE2D_DESC));
}

std::unique_ptr<ICRCContainable> CRCID3D11Texture2DFactoryL0_0::Create(IDESC &desc) const
{
    CRC_TEXTURE2D_DESC* textureDesc = CRC::As<CRC_TEXTURE2D_DESC>(&desc);
    if (!textureDesc) return nullptr;

    std::unique_ptr<CRCID3D11Texture2D> texture = std::make_unique<CRCID3D11Texture2D>();

    D3D11_SUBRESOURCE_DATA* initialData = nullptr;
    if (textureDesc->SysMem()) initialData = &textureDesc->InitialData();

    if (!textureDesc->d3d11Device_) throw std::runtime_error("Device not set.");

    HRESULT hr = textureDesc->d3d11Device_->CreateTexture2D
    (
        &textureDesc->Desc(), initialData, texture->Get().GetAddressOf()
    );
    if (FAILED(hr)) return nullptr;

    return texture;
}

Microsoft::WRL::ComPtr<ID3D11Resource> &CRCID3D11Texture2D::GetResource()
{
    Microsoft::WRL::ComPtr<ID3D11Resource> resource;
    d3d11Texture2D_.As(&resource);

    return resource;
}

HRESULT CRCID3D11Texture2D::GetType(D3D11_RESOURCE_DIMENSION &type)
{
    d3d11Texture2D_->GetType(&type);
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

CRCTexutre2DAttached::CRCTexutre2DAttached(D3D11_TEXTURE2D_DESC &desc)
{
    desc_ = desc;
}

CRCTexutre2DAttached::~CRCTexutre2DAttached()
{
    if (cudaArray_) Unassign();
}

HRESULT CRCTexutre2DAttached::GetType(D3D11_RESOURCE_DIMENSION &type)
{
    type = D3D11_RESOURCE_DIMENSION::D3D11_RESOURCE_DIMENSION_TEXTURE2D;
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
}
