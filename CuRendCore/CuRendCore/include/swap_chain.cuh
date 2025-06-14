﻿#pragma once

#include "CuRendCore/include/config.h"
#include "WinAppCore/include/WACore.h"

#include "CuRendCore/include/factory.h"
#include "CuRendCore/include/texture.cuh"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

class CRC_API CRC_SWAP_CHAIN_DESC : public IDESC
{
private:
    DXGI_SWAP_CHAIN_DESC desc_ = {};

public:
    CRC_SWAP_CHAIN_DESC
    (
        Microsoft::WRL::ComPtr<ID3D11Device>& d3d11Device, Microsoft::WRL::ComPtr<IDXGISwapChain>& d3d11SwapChain
    ) : d3d11Device_(d3d11Device), d3d11SwapChain_(d3d11SwapChain) {}
    ~CRC_SWAP_CHAIN_DESC() override = default;

    Microsoft::WRL::ComPtr<ID3D11Device>& d3d11Device_;
    Microsoft::WRL::ComPtr<IDXGISwapChain>& d3d11SwapChain_;
    DXGI_SWAP_CHAIN_DESC& GetDxgiDesc() { return desc_; }
};

class CRC_API ICRCSwapChain
{
public:
    virtual ~ICRCSwapChain() = default;
    virtual Microsoft::WRL::ComPtr<IDXGISwapChain>& GetD3D11SwapChain() = 0;

    virtual HRESULT GetBuffer(UINT buffer, ICRCTexture2D*& texture) = 0;
    virtual HRESULT ResizeBuffers
    (
        UINT bufferCount, UINT width, UINT height, DXGI_FORMAT newFormat, UINT swapChainFlags
    ) = 0;

    virtual HRESULT Present(UINT syncInterval, UINT flags) = 0;

    virtual HRESULT GetDesc(DXGI_SWAP_CHAIN_DESC *pDesc) = 0;
};

class CRC_API CRCSwapChainFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCSwapChainFactoryL0_0() override = default;
    virtual std::unique_ptr<WACore::IContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCSwapChain : public WACore::IContainable, public ICRCSwapChain
{
private:
Microsoft::WRL::ComPtr<ID3D11Device>& d3d11Device_;
    Microsoft::WRL::ComPtr<IDXGISwapChain>& d3d11SwapChain_;

    UINT bufferCount_;
    DXGI_RATIONAL refreshRate_;
    UINT frameIndex_ = 0;
    bool presentExecuted_ = false;

    std::vector<cudaGraphicsResource_t> cudaResources_;
    std::vector<ICRCTexture2D*> backSurfaces_;

    ICRCTexture2D* backBuffer_ = nullptr;

public:
    CRCSwapChain
    (
        Microsoft::WRL::ComPtr<ID3D11Device>& d3d11Device,
        Microsoft::WRL::ComPtr<IDXGISwapChain>& d3d11SwapChain,
        const DXGI_SWAP_CHAIN_DESC& desc
    );

    ~CRCSwapChain() override;

    // Delete copy constructor and copy assignment
    CRCSwapChain(const CRCSwapChain&) = delete;
    CRCSwapChain& operator=(const CRCSwapChain&) = delete;

    Microsoft::WRL::ComPtr<IDXGISwapChain>& GetD3D11SwapChain() override { return d3d11SwapChain_; }

    HRESULT GetBuffer(UINT buffer, ICRCTexture2D*& texture) override;
    HRESULT ResizeBuffers
    (
        UINT bufferCount, UINT width, UINT height, DXGI_FORMAT newFormat, UINT swapChainFlags
    ) override;

    HRESULT Present(UINT syncInterval, UINT flags) override;

    HRESULT GetDesc(DXGI_SWAP_CHAIN_DESC* pDesc) override;
};

class CRC_API CRCIDXGISwapChainFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCIDXGISwapChainFactoryL0_0() override = default;
    virtual std::unique_ptr<WACore::IContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCIDXGISwapChain : public WACore::IContainable, public ICRCSwapChain
{
private:
    Microsoft::WRL::ComPtr<IDXGISwapChain>& d3d11SwapChain_;
    ICRCTexture2D* backBuffer_ = nullptr;

public:
    CRCIDXGISwapChain(Microsoft::WRL::ComPtr<IDXGISwapChain>& d3d11SwapChain);
    ~CRCIDXGISwapChain() override;

    Microsoft::WRL::ComPtr<IDXGISwapChain>& GetD3D11SwapChain() override { return d3d11SwapChain_; }

    HRESULT GetBuffer(UINT buffer, ICRCTexture2D*& texture) override;
    HRESULT ResizeBuffers
    (
        UINT bufferCount, UINT width, UINT height, DXGI_FORMAT newFormat, UINT swapChainFlags
    ) override;

    HRESULT Present(UINT syncInterval, UINT flags) override;

    HRESULT GetDesc(DXGI_SWAP_CHAIN_DESC* pDesc) override;
};