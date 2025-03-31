#include "CuRendCore/include/pch.h"
#include "CuRendCore/include/funcs.cuh"

#include "CuRendCore/include/rtv.cuh"
#include "CuRendCore/include/texture.cuh"

std::unique_ptr<WACore::IContainable> CRCRenderTargetViewFactoryL0_0::Create(IDESC &desc) const
{
    CRC_RENDER_TARGET_VIEW_DESC* rtvDesc = WACore::As<CRC_RENDER_TARGET_VIEW_DESC>(&desc);
    if (!rtvDesc)
    {
#ifndef NDEBUG
        CRC::CoutWarning
        (
            "Failed to create render target view from desc. Desc is not CRC_RENDER_TARGET_VIEW_DESC."
        );
#endif
        return nullptr;
    }

    if (!rtvDesc->resource_)
    {
#ifndef NDEBUG
        CRC::CoutWarning
        (
            "Failed to create render target view from desc. Resource is nullptr."
        );
#endif
        return nullptr;
    }

    {
        WACore::RevertCast<ICRCTexture2D, WACore::IContainable> texture(rtvDesc->resource_);
        if (!texture())
        {
#ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to create render target view from desc. Resource is not ICRCTexture2D."
            );
#endif
            return nullptr;
        }

        D3D11_TEXTURE2D_DESC desc;
        texture()->GetDesc(&desc);

        if (!(desc.BindFlags & D3D11_BIND_RENDER_TARGET))
        {
#ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to create render target view from desc. Texture2D is not bindable as render target."
            );
#endif
            return nullptr;
        }
    }

    std::unique_ptr<CRCRenderTargetView> rtv = std::make_unique<CRCRenderTargetView>
    (
        rtvDesc->resource_, rtvDesc->desc_
    );

    return rtv;
}


CRCRenderTargetView::CRCRenderTargetView
(
    std::unique_ptr<WACore::IContainable> &resource, D3D11_RENDER_TARGET_VIEW_DESC& desc
) : resource_(resource)
{
    desc_ = desc;

#ifndef NDEBUG
    CRC::Cout("Created render target view from desc.");
#endif
}

CRCRenderTargetView::~CRCRenderTargetView()
{
#ifndef NDEBUG
    CRC::Cout("Destroyed render target view.");
#endif
}

const void CRCRenderTargetView::GetDesc(D3D11_RENDER_TARGET_VIEW_DESC *dst)
{
    std::memcpy(dst, &desc_, sizeof(D3D11_RENDER_TARGET_VIEW_DESC));
}

std::unique_ptr<WACore::IContainable> CRCID3D11RenderTargetViewFactoryL0_0::Create(IDESC &desc) const
{
    CRC_RENDER_TARGET_VIEW_DESC* rtvDesc = WACore::As<CRC_RENDER_TARGET_VIEW_DESC>(&desc);
    if (!rtvDesc)
    {
#ifndef NDEBUG
        CRC::CoutWarning
        (
            "Failed to create render target view from desc. Desc is not CRC_RENDER_TARGET_VIEW_DESC."
        );
#endif
        return nullptr;
    }

    if (!rtvDesc->d3d11Device_)
    {
#ifndef NDEBUG
        CRC::CoutWarning
        (
            "Failed to create render target view from desc. D3D11 device is nullptr."
        );
#endif
        return nullptr;
    }

    std::unique_ptr<CRCID3D11RenderTargetView> rtv;
    {
        WACore::RevertCast<CRCID3D11Texture2D, WACore::IContainable> texture(rtvDesc->resource_);
        if (!texture())
        {
    #ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to create render target view from desc. Resource is not CRCID3D11Texture2D."
            );
    #endif
            return nullptr;
        }

        rtv = std::make_unique<CRCID3D11RenderTargetView>();

        HRESULT hr = rtvDesc->d3d11Device_->CreateRenderTargetView
        (
            texture()->Get().Get(), &rtvDesc->desc_, rtv->Get().GetAddressOf()
        );
        if (FAILED(hr))
        {
    #ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to create render target view from desc. D3D11Device CreateRenderTargetView failed."
            );
    #endif
            return nullptr;
        }
    }

    return rtv;
}

CRCID3D11RenderTargetView::CRCID3D11RenderTargetView()
{
#ifndef NDEBUG
    CRC::Cout("Created D3D11 render target view from desc.");
#endif
}

CRCID3D11RenderTargetView::~CRCID3D11RenderTargetView()
{
#ifndef NDEBUG
    CRC::Cout("Destroyed D3D11 render target view.");
#endif
}

const void CRCID3D11RenderTargetView::GetDesc(D3D11_RENDER_TARGET_VIEW_DESC *dst)
{
    d3d11RTV_->GetDesc(dst);
}
