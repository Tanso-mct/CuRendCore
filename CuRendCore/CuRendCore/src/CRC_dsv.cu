#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_dsv.cuh"
#include "CRC_texture.cuh"

std::unique_ptr<ICRCContainable> CRCDepthStencilViewFactoryL0_0::Create(IDESC &desc) const
{
    CRC_DEPTH_STENCIL_VIEW_DESC *dsvDesc = CRC::As<CRC_DEPTH_STENCIL_VIEW_DESC>(&desc);
    if (!dsvDesc)
    {
#ifndef NDEBUG
        CRC::CoutWarning
        (
            "Failed to create depth stencil view from desc. Desc is not CRC_DEPTH_STENCIL_VIEW_DESC."
        );
#endif
        return nullptr;
    }

    if (!dsvDesc->resource_)
    {
#ifndef NDEBUG
        CRC::CoutWarning
        (
            "Failed to create depth stencil view from desc. Resource is nullptr."
        );
#endif
        return nullptr;
    }

    {
        CRCTransCastUnique<ICRCTexture2D, ICRCContainable> texture(dsvDesc->resource_);
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

        if (!(desc.BindFlags & D3D11_BIND_DEPTH_STENCIL))
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

    std::unique_ptr<CRCDepthStencilView> dsv = std::make_unique<CRCDepthStencilView>
    (
        dsvDesc->resource_, dsvDesc->desc_
    );

#ifndef NDEBUG
    CRC::Cout("Created depth stencil view from desc.");
#endif

    return dsv;
}

CRCDepthStencilView::CRCDepthStencilView
(
    std::unique_ptr<ICRCContainable> &resource, D3D11_DEPTH_STENCIL_VIEW_DESC &desc
) : resource_(resource)
{
    desc_ = desc;
}

CRCDepthStencilView::~CRCDepthStencilView()
{
#ifndef NDEBUG
    CRC::Cout("Destroyed depth stencil view.");
#endif
}

const void CRCDepthStencilView::GetDesc(D3D11_DEPTH_STENCIL_VIEW_DESC *dst)
{
    std::memcpy(dst, &desc_, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));
}

std::unique_ptr<ICRCContainable> CRCID3D11DepthStencilViewFactoryL0_0::Create(IDESC &desc) const
{
    CRC_DEPTH_STENCIL_VIEW_DESC *dsvDesc = CRC::As<CRC_DEPTH_STENCIL_VIEW_DESC>(&desc);
    if (!dsvDesc)
    {
#ifndef NDEBUG
        CRC::CoutWarning
        (
            "Failed to create depth stencil view from desc. Desc is not CRC_DEPTH_STENCIL_VIEW_DESC."
        );
#endif
        return nullptr;
    }

    if (!dsvDesc->d3d11Device_)
    {
#ifndef NDEBUG
        CRC::CoutWarning
        (
            "Failed to create depth stencil view from desc. D3D11 device is nullptr."
        );
#endif
        return nullptr;
    }

    std::unique_ptr<CRCID3D11DepthStencilView> dsv;
    {
        CRCTransCastUnique<CRCID3D11Texture2D, ICRCContainable> texture(dsvDesc->resource_);
        if (!texture())
        {
    #ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to create depth stencil view from desc. Resource is not CRCID3D11Texture2D."
            );
    #endif
            return nullptr;
        }

        dsv = std::make_unique<CRCID3D11DepthStencilView>();

        HRESULT hr = dsvDesc->d3d11Device_->CreateDepthStencilView
        (
            texture()->Get().Get(), &dsvDesc->desc_, dsv->Get().GetAddressOf()
        );
        if (FAILED(hr))
        {
    #ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to create depth stencil view from desc. D3D11Device CreateDepthStencilView failed."
            );
    #endif
            return nullptr;
        }
    }

#ifndef NDEBUG
    CRC::Cout("Created D3D11 depth stencil view from desc.");
#endif

    return dsv;
}

CRCID3D11DepthStencilView::~CRCID3D11DepthStencilView()
{
#ifndef NDEBUG
    CRC::Cout("Destroyed D3D11 depth stencil view.");
#endif
}

const void CRCID3D11DepthStencilView::GetDesc(D3D11_DEPTH_STENCIL_VIEW_DESC *dst)
{
    d3d11DSV->GetDesc(dst);
}
