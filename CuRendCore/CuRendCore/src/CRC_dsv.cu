#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_dsv.cuh"

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

    std::unique_ptr<CRCDepthStencilView> dsv = std::make_unique<CRCDepthStencilView>
    (
        dsvDesc->resource_, dsvDesc->desc_
    );
    return dsv;
}

CRCDepthStencilView::CRCDepthStencilView
(
    std::unique_ptr<ICRCContainable> &resource, D3D11_DEPTH_STENCIL_VIEW_DESC &desc
) : resource_(resource)
{
    desc_ = desc;
}

const void CRCDepthStencilView::GetDesc(D3D11_DEPTH_STENCIL_VIEW_DESC *dst)
{
    std::memcpy(dst, &desc_, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));
}

const void CRCID3D11DepthStencilView::GetDesc(D3D11_DEPTH_STENCIL_VIEW_DESC *dst)
{
    d3d11DSV->GetDesc(dst);
}
