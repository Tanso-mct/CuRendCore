﻿#pragma once

#include "CuRendCore/include/config.h"
#include "WinAppCore/include/WACore.h"

#include "CuRendCore/include/view.cuh"
#include "CuRendCore/include/factory.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>

class ICRCCreatable;

class CRC_API CRC_SHADER_RESOURCE_VIEW_DESC : public IDESC
{
public:
    CRC_SHADER_RESOURCE_VIEW_DESC() = delete;
    CRC_SHADER_RESOURCE_VIEW_DESC
    (
        Microsoft::WRL::ComPtr<ID3D11Device>& device, std::unique_ptr<WACore::IContainable>& resource
    ) : d3d11Device_(device), resource_(resource) {}
    ~CRC_SHADER_RESOURCE_VIEW_DESC() override = default;

    Microsoft::WRL::ComPtr<ID3D11Device>& d3d11Device_;

    D3D11_SHADER_RESOURCE_VIEW_DESC desc_ = {};
    std::unique_ptr<WACore::IContainable>& resource_;
};

class CRC_API ICRCShaderResourceView
{
public:
    virtual ~ICRCShaderResourceView() = default;
    virtual const void GetDesc(D3D11_SHADER_RESOURCE_VIEW_DESC* dst) = 0;
};

class CRC_API CRCShaderResourceViewFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCShaderResourceViewFactoryL0_0() override = default;
    virtual std::unique_ptr<WACore::IContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCShaderResourceView 
: public WACore::IContainable, public ICRCView, public ICRCShaderResourceView
{
private:
    D3D11_SHADER_RESOURCE_VIEW_DESC desc_ = {};
    std::unique_ptr<WACore::IContainable>& resource_;

public:
    CRCShaderResourceView() = delete;
    CRCShaderResourceView(std::unique_ptr<WACore::IContainable>& resource, D3D11_SHADER_RESOURCE_VIEW_DESC& desc);
    virtual ~CRCShaderResourceView() override;
    
    // ICRCView
    virtual std::unique_ptr<WACore::IContainable>& GetResource() override { return resource_; }

    // ICRCShaderResourceView
    virtual const void GetDesc(D3D11_SHADER_RESOURCE_VIEW_DESC* dst) override;
};

class CRC_API CRCID3D11ShaderResourceViewFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCID3D11ShaderResourceViewFactoryL0_0() override = default;
    virtual std::unique_ptr<WACore::IContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCID3D11ShaderResourceView 
: public WACore::IContainable, public ICRCView, public ICRCShaderResourceView
{
private:
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> d3d11SRV_;
    std::unique_ptr<WACore::IContainable> emptyResource_ = nullptr;

public:
    CRCID3D11ShaderResourceView();
    virtual ~CRCID3D11ShaderResourceView() override;

    // ICRCView
    virtual std::unique_ptr<WACore::IContainable>& GetResource() override { return emptyResource_; }

    // ICRCShaderResourceView
    virtual const void GetDesc(D3D11_SHADER_RESOURCE_VIEW_DESC* dst) override;

    virtual Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>& Get() { return d3d11SRV_; }
};