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

CRCTexture2D::CRCTexture2D()
{
    dMem_ = std::make_unique<CRCDeviceMem>();
}

CRCTexture2D::CRCTexture2D(CRC_TEXTURE2D_DESC& desc)
{
    dMem_ = std::make_unique<CRCDeviceMem>();
    dMem_->Malloc
    (
        CRC::GetBytesPerPixel(desc.Format()) * desc.Width() * desc.Height(), 
        CRC::GetBytesPerPixel(desc.Format()) * desc.Width(), 
        CRC::GetBytesPerPixel(desc.Format()) * desc.Width() * desc.Height()
    );

    if (desc.SysMem())
    {
        CRC::CheckCuda(cudaMemcpy
        (
            dMem_.get(), desc.SysMem(), dMem_->GetByteWidth(), cudaMemcpyHostToDevice
        ));
    }

    desc_ = desc.Desc();
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

const UINT &CRCID3D11Texture2D::GetByteWidth() const
{
    D3D11_TEXTURE2D_DESC desc;
    d3d11Texture2D_->GetDesc(&desc);

    return CRC::GetBytesPerPixel(desc.Format) * desc.Width * desc.Height;
}

const UINT &CRCID3D11Texture2D::GetPitch() const
{
    D3D11_TEXTURE2D_DESC desc;
    d3d11Texture2D_->GetDesc(&desc);

    return CRC::GetBytesPerPixel(desc.Format) * desc.Width;
}

const UINT &CRCID3D11Texture2D::GetSlicePitch() const
{
    D3D11_TEXTURE2D_DESC desc;
    d3d11Texture2D_->GetDesc(&desc);

    return CRC::GetBytesPerPixel(desc.Format) * desc.Width * desc.Height;
}

const void CRCID3D11Texture2D::GetDesc(D3D11_TEXTURE2D_DESC *dst)
{
    d3d11Texture2D_->GetDesc(dst);
}
