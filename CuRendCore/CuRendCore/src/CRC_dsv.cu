#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_dsv.cuh"

std::unique_ptr<ICRCResource> &CRCDepthStencilView::GetResource()
{
    return resource;
}

std::unique_ptr<ICRCResource> &CRCID3D11DepthStencilView::GetResource()
{
    return emptyResource;
}
