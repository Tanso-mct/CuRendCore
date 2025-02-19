#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_rtv.cuh"

std::unique_ptr<ICRCResource> &CRCRenderTargetView::GetResource()
{
    return resource_;
}

std::unique_ptr<ICRCResource> &CRCID3D11RenderTargetView::GetResource()
{
    return emptyResource_;
}
