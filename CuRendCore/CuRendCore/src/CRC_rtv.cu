#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_rtv.cuh"

#include "CRC_container.h"

std::unique_ptr<ICRCContainable> CRCRenderTargetViewFactoryL0_0::Create(IDESC &desc) const
{
    CRC_RENDER_TARGET_VIEW_DESC* rtvDesc = CRC::As<CRC_RENDER_TARGET_VIEW_DESC>(&desc);
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

    std::unique_ptr<CRCRenderTargetView> rtv = std::make_unique<CRCRenderTargetView>
    (
        rtvDesc->resource_, rtvDesc->desc_
    );
    return rtv;
}


CRCRenderTargetView::CRCRenderTargetView
(
    std::unique_ptr<ICRCContainable> &resource, D3D11_RENDER_TARGET_VIEW_DESC& desc
) : resource_(resource)
{
    desc_ = desc;
}

const void CRCRenderTargetView::GetDesc(D3D11_RENDER_TARGET_VIEW_DESC *dst)
{
    std::memcpy(dst, &desc_, sizeof(D3D11_RENDER_TARGET_VIEW_DESC));
}

const void CRCID3D11RenderTargetView::GetDesc(D3D11_RENDER_TARGET_VIEW_DESC *dst)
{
    d3d11RTV_->GetDesc(dst);
}