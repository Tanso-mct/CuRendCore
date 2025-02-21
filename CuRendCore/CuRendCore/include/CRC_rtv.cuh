#pragma once

#include "CRC_config.h"
#include "CRC_container.h"
#include "CRC_view.cuh"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>

class CRC_API ICRCRenderTargetView
{
public:
    virtual ~ICRCRenderTargetView() = default;
};

class CRC_API CRCRenderTargetView : public ICRCContainable, public ICRCView, public ICRCRenderTargetView
{
private:
    std::unique_ptr<ICRCResource>& resource_;

public:
    ~CRCRenderTargetView() override = default;

    virtual std::unique_ptr<ICRCResource>& GetResource() override;
};

class CRC_API CRCID3D11RenderTargetView : public ICRCContainable, public ICRCView, public ICRCRenderTargetView
{
private:
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> d3d11RTV_;
    std::unique_ptr<ICRCResource> emptyResource_ = nullptr;

public:
    ~CRCID3D11RenderTargetView() override = default;

    virtual std::unique_ptr<ICRCResource>& GetResource() override;
};