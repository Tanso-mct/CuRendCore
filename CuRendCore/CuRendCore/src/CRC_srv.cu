#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_srv.cuh"
#include "CRC_texture.cuh"

std::unique_ptr<ICRCContainable> CRCShaderResourceViewFactoryL0_0::Create(IDESC &desc) const
{
    CRC_SHADER_RESOURCE_VIEW_DESC* srvDesc = CRC::As<CRC_SHADER_RESOURCE_VIEW_DESC>(&desc);
    if (!srvDesc)
    {
#ifndef NDEBUG
        CRC::CoutWarning
        (
            "Failed to create shader resource view from desc. Desc is not CRC_SHADER_RESOURCE_VIEW_DESC."
        );
#endif
        return nullptr;
    }

    std::unique_ptr<CRCShaderResourceView> srv = std::make_unique<CRCShaderResourceView>
    (
        srvDesc->resource_, srvDesc->desc_
    );
    return srv;
}

CRCShaderResourceView::CRCShaderResourceView
(
    std::unique_ptr<ICRCContainable> &resource, D3D11_SHADER_RESOURCE_VIEW_DESC &desc
) : resource_(resource)
{
    desc_ = desc;
}

const void CRCShaderResourceView::GetDesc(D3D11_SHADER_RESOURCE_VIEW_DESC *dst)
{
    std::memcpy(dst, &desc_, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
}

std::unique_ptr<ICRCContainable> CRCID3D11ShaderResourceViewFactoryL0_0::Create(IDESC &desc) const
{
    CRC_SHADER_RESOURCE_VIEW_DESC* srvDesc = CRC::As<CRC_SHADER_RESOURCE_VIEW_DESC>(&desc);
    if (!srvDesc)
    {
#ifndef NDEBUG
        CRC::CoutWarning
        (
            "Failed to create shader resource view from desc. Desc is not CRC_SHADER_RESOURCE_VIEW_DESC."
        );
#endif
        return nullptr;
    }

    if (!srvDesc->d3d11Device_)
    {
#ifndef NDEBUG
        CRC::CoutWarning
        (
            "Failed to create shader resource view from desc. D3D11 device is nullptr."
        );
#endif
        return nullptr;
    }

    std::unique_ptr<CRCID3D11ShaderResourceView> srv;
    {
        CRCTransCastUnique<CRCID3D11Texture2D, ICRCContainable> texture(srvDesc->resource_);
        if (!texture())
        {
    #ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to create shader resource view from desc. Resource is not CRCID3D11Texture2D."
            );
    #endif
            return nullptr;
        }

        srv = std::make_unique<CRCID3D11ShaderResourceView>();

        HRESULT hr = srvDesc->d3d11Device_->CreateShaderResourceView
        (
            texture()->Get().Get(), &srvDesc->desc_, srv->Get().GetAddressOf()
        );
        if (FAILED(hr))
        {
    #ifndef NDEBUG
            CRC::CoutWarning
            (
                "Failed to create shader resource view from desc. D3D11Device CreateShaderResourceView failed."
            );
    #endif
            return nullptr;
        }
    }

    return srv;
}

const void CRCID3D11ShaderResourceView::GetDesc(D3D11_SHADER_RESOURCE_VIEW_DESC *dst)
{
    d3d11SRV_->GetDesc(dst);
}
