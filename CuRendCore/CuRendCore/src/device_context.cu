#include "CuRendCore/include/pch.h"
#include "CuRendCore/include/funcs.cuh"

#include "CuRendCore/include/container.h"
#include "CuRendCore/include/resource.cuh"
#include "CuRendCore/include/memory.cuh"
#include "CuRendCore/include/buffer.cuh"
#include "CuRendCore/include/texture.cuh"
#include "CuRendCore/include/device_context.cuh"

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
    std::unique_ptr<ICRCContainable> &resource, 
    UINT subresource, 
    D3D11_MAP mapType, 
    UINT mapFlags, 
    D3D11_MAPPED_SUBRESOURCE *mappedResource
){
    {
        WACore::RevertCast<ICRCMemory, ICRCContainable> memory(resource);
        if (!memory())
        {
#ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to map resource.",
                "This resource is not CRC resource."
            );
#endif
            return E_FAIL;
        }

        if (!memory()->IsCpuAccessible())
        {
#ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to map resource.",
                "This resource is not CPU accessible."
            );
#endif
            return E_FAIL;
        }

        memory()->SendDeviceToHost();
        mappedResource->pData = memory()->GetHostPtr();
        mappedResource->RowPitch = memory()->GetRowPitch();
        mappedResource->DepthPitch = memory()->GetDepthPitch();
    }

    return S_OK;
}

void CRCImmediateContext::Unmap
(
    std::unique_ptr<ICRCContainable> &resource, 
    UINT subresource
){
    {
        WACore::RevertCast<ICRCMemory, ICRCContainable> memory(resource);
        if (!memory())
        {
#ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to unmap resource.",
                "This resource is not CRC resource."
            );
#endif
            return;
        }

        if (!memory()->IsCpuAccessible())
        {
#ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to unmap resource.",
                "This resource is not CPU accessible."
            );
#endif
            return;
        }

        HRESULT hr = memory()->SendHostToDevice();
        if (FAILED(hr))
        {
#ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to unmap resource.",
                "SendHostToDevice failed."
            );
#endif
            return;
        }
    }
}

void CRCImmediateContext::UpdateSubresource
(
    std::unique_ptr<ICRCContainable> &dst, 
    const void *src, 
    UINT srcByteWidth
){
    {
        WACore::RevertCast<ICRCMemory, ICRCContainable> memory(dst);
        if (!memory())
        {
#ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to update subresource.",
                "This resource is not CRC resource."
            );
#endif
            return;
        }

        if (!memory()->IsCpuAccessible())
        {
#ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to update subresource.",
                "This resource is not CPU accessible."
            );
#endif
            return;
        }

        memory()->SendHostToDevice(src, srcByteWidth);
    }
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
    std::unique_ptr<ICRCContainable> &resource, 
    UINT subresource, 
    D3D11_MAP mapType, 
    UINT mapFlags, 
    D3D11_MAPPED_SUBRESOURCE *mappedResource
){
    {
        WACore::RevertCast<ICRCID3D11Resource, ICRCContainable> d3d11Resource(resource);
        if (!d3d11Resource())
        {
#ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to map resource.",
                "This resource is not ID3D11 resource."
            );
#endif
            return E_FAIL;
        }

        HRESULT hr = d3d11DeviceContext->Map
        (
            d3d11Resource()->GetResource().Get(), subresource, mapType, mapFlags, mappedResource
        );
        if (FAILED(hr))
        {
#ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to map resource.",
                "D3D11DeviceContext Map failed."
            );
#endif
            return hr;
        }
    }

    return S_OK;
}

void CRCID3D11Context::Unmap
(
    std::unique_ptr<ICRCContainable> &resource, 
    UINT subresource
){
    {
        WACore::RevertCast<ICRCID3D11Resource, ICRCContainable> d3d11Resource(resource);
        if (!d3d11Resource())
        {
#ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to unmap resource.",
                "This resource is not ID3D11 resource."
            );
#endif
            return;
        }

        d3d11DeviceContext->Unmap(d3d11Resource()->GetResource().Get(), subresource);
    }
}

void CRCID3D11Context::UpdateSubresource
(
    std::unique_ptr<ICRCContainable> &dst, 
    const void *src, 
    UINT srcByteWidth
){
    {
        WACore::RevertCast<ICRCID3D11Resource, ICRCContainable> d3d11Resource(dst);
        if (!d3d11Resource())
        {
#ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to update subresource.",
                "This resource is not ID3D11 resource."
            );
#endif
            return;
        }

        d3d11DeviceContext->UpdateSubresource(d3d11Resource()->GetResource().Get(), 0, nullptr, src, 0, 0);
    }
}
