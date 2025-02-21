#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_texture.cuh"

std::unique_ptr<ICRCContainable> CRCTexture2DFactory::Create(IDESC &desc) const
{
    CRC_TEXTURE_DESC* textureDesc = CRC::As<CRC_TEXTURE_DESC>(&desc);
    if (!textureDesc) return nullptr;

    std::unique_ptr<CRCTexture2D> texture = std::make_unique<CRCTexture2D>();
    texture->dMem = std::make_unique<CRCDeviceMem>();

    texture->dMem->Malloc
    (
        CRC::GetBytesPerPixel(textureDesc->Format()) * textureDesc->Width() * textureDesc->Height(),
        CRC::GetBytesPerPixel(textureDesc->Format()) * textureDesc->Width(),
        0
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

std::unique_ptr<ICRCContainable> CRCID3D11Texture2DFactory::Create(IDESC &desc) const
{
    CRC_TEXTURE_DESC* textureDesc = CRC::As<CRC_TEXTURE_DESC>(&desc);
    if (!textureDesc) return nullptr;

    std::unique_ptr<CRCID3D11Texture2D> texture = std::make_unique<CRCID3D11Texture2D>();

    D3D11_SUBRESOURCE_DATA* initialData = nullptr;
    if (textureDesc->SysMem()) initialData = &textureDesc->InitialData();

    HRESULT hr = textureDesc->device_->CreateTexture2D
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

const D3D11_TEXTURE2D_DESC &CRCID3D11Texture2D::GetDesc()
{
    d3d11Texture2D->GetDesc(&desc_);
    return desc_;
}
