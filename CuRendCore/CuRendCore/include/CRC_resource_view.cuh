#pragma once

#include "CRC_config.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class ICRCView
{
public:
    virtual ~ICRCView() = default;
};

class CRCRenderTargetView : public ICRCView
{
public:
    virtual ~CRCRenderTargetView() = default;
};

class CRCID3D11RenderTargetView : public ICRCView
{
private:
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> d3d11RTV;

public:
    ~CRCID3D11RenderTargetView() override = default;
};

class CRCShaderResourceView : public ICRCView
{
public:
    virtual ~CRCShaderResourceView() = default;
};

class CRCID3D11ShaderResourceView : public ICRCView
{
private:
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> d3d11SRV;

public:
    ~CRCID3D11ShaderResourceView() override = default;
};