#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_srv.cuh"

std::unique_ptr<ICRCResource> &CRCShaderResourceView::GetResource()
{
    return resource;
}

std::unique_ptr<ICRCResource> &CRCID3D11ShaderResourceView::GetResource()
{
    return emptyResource;
}
