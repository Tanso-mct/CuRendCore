#pragma once

#include "CRC_config.h"
#include "CRC_factory.h"
#include "CRC_texture.cuh"

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
    Microsoft::WRL::ComPtr<IDXGISwapChain>& d3d11SwapChain_;

public:
    CRC_SWAP_CHAIN_DESC(Microsoft::WRL::ComPtr<IDXGISwapChain>& d3d11SwapChain) : d3d11SwapChain_(d3d11SwapChain) {}
    ~CRC_SWAP_CHAIN_DESC() override = default;

    Microsoft::WRL::ComPtr<IDXGISwapChain>& GetD3D11SwapChain() { return d3d11SwapChain_; }

    UINT& BufferCount() { return desc_.BufferCount; }
    DXGI_USAGE& BufferUsage() { return desc_.BufferUsage; }
    DXGI_RATIONAL& RefreshRate() { return desc_.BufferDesc.RefreshRate; }
    DXGI_SWAP_EFFECT& SwapEffect() { return desc_.SwapEffect; }
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
};

class CRC_API CRCSwapChainFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCSwapChainFactoryL0_0() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCSwapChain : public ICRCContainable, public ICRCSwapChain
{
private:
    Microsoft::WRL::ComPtr<IDXGISwapChain>& d3d11SwapChain_;

    UINT bufferCount_;
    const DXGI_USAGE bufferUsage_;
    const DXGI_RATIONAL refreshRate_;
    const DXGI_SWAP_EFFECT swapEffect_;

    std::vector<cudaGraphicsResource_t> cudaResources_;
    UINT frameIndex_ = 0;

    std::unique_ptr<CRCTexture2D> backBuffer_ = nullptr;

public:
    CRCSwapChain
    (
        Microsoft::WRL::ComPtr<IDXGISwapChain>& d3d11SwapChain,
        UINT bufferCount, DXGI_USAGE bufferUsage, DXGI_RATIONAL refreshRate, DXGI_SWAP_EFFECT swapEffect
    );

    ~CRCSwapChain() override;

    Microsoft::WRL::ComPtr<IDXGISwapChain>& GetD3D11SwapChain() override { return d3d11SwapChain_; }

    HRESULT GetBuffer(UINT buffer, ICRCTexture2D*& texture) override;
    HRESULT ResizeBuffers
    (
        UINT bufferCount, UINT width, UINT height, DXGI_FORMAT newFormat, UINT swapChainFlags
    ) override;

    HRESULT Present(UINT syncInterval, UINT flags) override;
};

class CRC_API CRCIDXGISwapChainFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCIDXGISwapChainFactoryL0_0() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCIDXGISwapChain : public ICRCContainable, public ICRCSwapChain
{
private:
    Microsoft::WRL::ComPtr<IDXGISwapChain>& d3d11SwapChain_;

public:
    CRCIDXGISwapChain(Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain) : d3d11SwapChain_(d3d11SwapChain) {}
    ~CRCIDXGISwapChain() override = default;

    Microsoft::WRL::ComPtr<IDXGISwapChain>& GetD3D11SwapChain() override { return d3d11SwapChain_; }

    HRESULT GetBuffer(UINT buffer, ICRCTexture2D*& texture) override;
    HRESULT ResizeBuffers
    (
        UINT bufferCount, UINT width, UINT height, DXGI_FORMAT newFormat, UINT swapChainFlags
    ) override;

    HRESULT Present(UINT syncInterval, UINT flags) override;
};