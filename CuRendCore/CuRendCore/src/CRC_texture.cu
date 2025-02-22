#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_texture.cuh"

std::unique_ptr<ICRCContainable> CRCTexture2DFactoryL0_0::Create(IDESC &desc) const
{
    CRC_TEXTURE2D_DESC* textureDesc = CRC::As<CRC_TEXTURE2D_DESC>(&desc);
    if (!textureDesc) return nullptr;

    std::unique_ptr<CRCTexture2D> texture = std::make_unique<CRCTexture2D>();
    texture->dMem = std::make_unique<CRCDeviceMem>();

    texture->dMem->Malloc
    (
        CRC::GetBytesPerPixel(textureDesc->Format()) * textureDesc->Width() * textureDesc->Height(),
        CRC::GetBytesPerPixel(textureDesc->Format()) * textureDesc->Width(),
        CRC::GetBytesPerPixel(textureDesc->Format()) * textureDesc->Width() * textureDesc->Height()
    );

    if (textureDesc->SysMem())
    {
        CRC::CheckCuda(cudaMemcpy
        (
            texture->dMem.get(), textureDesc->SysMem(), texture->dMem->GetByteWidth(), cudaMemcpyHostToDevice
        ));
    }

    texture->desc_ = textureDesc->Desc();

    return texture;
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
        &textureDesc->Desc(), initialData, texture->d3d11Texture2D.GetAddressOf()
    );
    if (FAILED(hr)) return nullptr;

    return texture;
}

Microsoft::WRL::ComPtr<ID3D11Resource> &CRCID3D11Texture2D::GetResource()
{
    Microsoft::WRL::ComPtr<ID3D11Resource> resource;
    d3d11Texture2D.As(&resource);

    return resource;
}

const UINT &CRCID3D11Texture2D::GetByteWidth() const
{
    D3D11_TEXTURE2D_DESC desc;
    d3d11Texture2D->GetDesc(&desc);

    return CRC::GetBytesPerPixel(desc.Format) * desc.Width * desc.Height;
}

const UINT &CRCID3D11Texture2D::GetPitch() const
{
    D3D11_TEXTURE2D_DESC desc;
    d3d11Texture2D->GetDesc(&desc);

    return CRC::GetBytesPerPixel(desc.Format) * desc.Width;
}

const UINT &CRCID3D11Texture2D::GetSlicePitch() const
{
    D3D11_TEXTURE2D_DESC desc;
    d3d11Texture2D->GetDesc(&desc);

    return CRC::GetBytesPerPixel(desc.Format) * desc.Width * desc.Height;
}

const void CRCID3D11Texture2D::GetDesc(D3D11_TEXTURE2D_DESC *dst)
{
    d3d11Texture2D->GetDesc(dst);
}
