#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_resource.cuh"
#include "CRC_device_context.cuh"

CRCImmediateContext::CRCImmediateContext()
{
#ifndef NDEBUG
    CRC::Cout("Created CRC immediate context.");
#endif
}

CRCImmediateContext::~CRCImmediateContext()
{
#ifndef NDEBUG
    CRC::Cout("Destroyed CRC immediate context.");
#endif
}

HRESULT CRCImmediateContext::Map
(
    std::unique_ptr<ICRCResource> &resource, 
    UINT subresource, 
    D3D11_MAP mapType, 
    UINT mapFlags, 
    D3D11_MAPPED_SUBRESOURCE *mappedResource
){
    UINT resType = 0;
    resource->GetResourceType(resType);

    return E_NOTIMPL;
}

void CRCImmediateContext::Unmap
(
    std::unique_ptr<ICRCResource> &resource, 
    UINT subresource
){
}

void CRCImmediateContext::UpdateSubresource
(
    std::unique_ptr<ICRCResource> &dst, 
    const void *src, 
    UINT srcByteWidth
){
}

CRCID3D11Context::CRCID3D11Context(ID3D11DeviceContext** d3d11DeviceContext)
: d3d11DeviceContext(*d3d11DeviceContext)
{
#ifndef NDEBUG
    CRC::Cout("Created CRC ID3D11 context.");
#endif
}

CRCID3D11Context::~CRCID3D11Context()
{
#ifndef NDEBUG
    CRC::Cout("Destroyed CRC ID3D11 context.");
#endif
}

HRESULT CRCID3D11Context::Map
(
    std::unique_ptr<ICRCResource> &resource, 
    UINT subresource, 
    D3D11_MAP mapType, 
    UINT mapFlags, 
    D3D11_MAPPED_SUBRESOURCE *mappedResource
){
    return E_NOTIMPL;
}

void CRCID3D11Context::Unmap
(
    std::unique_ptr<ICRCResource> &resource, 
    UINT subresource
){
}

void CRCID3D11Context::UpdateSubresource
(
    std::unique_ptr<ICRCResource> &dst, 
    const void *src, 
    UINT srcByteWidth
){
}
