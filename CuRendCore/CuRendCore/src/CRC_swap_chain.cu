#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_swap_chain.cuh"

std::unique_ptr<ICRCContainable> CRCSwapChainFactoryL0_0::Create(IDESC &desc) const
{
    CRC_SWAP_CHAIN_DESC* swapChainDesc = CRC::As<CRC_SWAP_CHAIN_DESC>(&desc);
    if (!swapChainDesc)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to create swap chain from desc. Desc is not CRC_SWAP_CHAIN_DESC.");
#endif
    }

    std::unique_ptr<CRCSwapChain> swapChain = std::make_unique<CRCSwapChain>
    (
        swapChainDesc->d3d11SwapChain_, swapChainDesc->GetDxgiDesc()
    );

    return swapChain;
}

CRCSwapChain::CRCSwapChain
(
    Microsoft::WRL::ComPtr<IDXGISwapChain> &d3d11SwapChain, 
    const DXGI_SWAP_CHAIN_DESC& desc
) 
: d3d11SwapChain_(d3d11SwapChain)
, bufferCount_(desc.BufferCount), refreshRate_(desc.BufferDesc.RefreshRate)
{
    CRC::RegisterCudaResources
    (
        cudaResources_, cudaGraphicsRegisterFlagsSurfaceLoadStore,
        bufferCount_, d3d11SwapChain_.Get()
    );

    backSurfaces_.resize(bufferCount_);
    for (int i = 0; i < bufferCount_; i++)
    {
        backSurfaces_[i] = CRC::CreateSurface2DFromCudaResource
        (
            cudaResources_[i], desc.BufferDesc.Width, desc.BufferDesc.Height, desc.BufferDesc.Format
        );
    }

    CRC::MapCudaResource(cudaResources_[frameIndex_]);
    backBuffer_ = backSurfaces_[frameIndex_].get();
}

CRCSwapChain::~CRCSwapChain()
{
    CRC::UnmapCudaResource(cudaResources_[frameIndex_]);
    backSurfaces_.clear();

    CRC::UnregisterCudaResourcesAtSwapChain(cudaResources_, d3d11SwapChain_, frameIndex_, bufferCount_);
    cudaResources_.clear();

    backBuffer_ = nullptr;
}

HRESULT CRCSwapChain::GetBuffer(UINT buffer, ICRCTexture2D*& texture)
{
    texture = backBuffer_;
    if (!texture) return E_FAIL;
    return S_OK;
}

HRESULT CRCSwapChain::ResizeBuffers
(
    UINT bufferCount, UINT width, UINT height, DXGI_FORMAT newFormat, UINT swapChainFlags
){
    CRC::UnmapCudaResource(cudaResources_[frameIndex_]);

    backSurfaces_.clear();

    CRC::UnregisterCudaResourcesAtSwapChain(cudaResources_, d3d11SwapChain_, frameIndex_, bufferCount_);
    cudaResources_.clear();

    HRESULT hr = d3d11SwapChain_->ResizeBuffers(bufferCount, width, height, newFormat, swapChainFlags);
    if (FAILED(hr))
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to resize buffers. IDXGISwapChain ResizeBuffers failed.");
#endif
    }

    frameIndex_ = 0;

    CRC::RegisterCudaResources
    (
        cudaResources_, cudaGraphicsRegisterFlagsSurfaceLoadStore, 
        bufferCount, d3d11SwapChain_.Get()
    );

    backSurfaces_.resize(bufferCount);
    for (int i = 0; i < bufferCount; i++)
    {
        backSurfaces_[i] = CRC::CreateSurface2DFromCudaResource
        (
            cudaResources_[i], width, height, newFormat
        );
    }

    CRC::MapCudaResource(cudaResources_[frameIndex_]);
    backBuffer_ = backSurfaces_[frameIndex_].get();

    return hr;
}

HRESULT CRCSwapChain::Present(UINT syncInterval, UINT flags)
{
    CRC::UnmapCudaResource(cudaResources_[frameIndex_]);

    HRESULT hr = d3d11SwapChain_->Present(syncInterval, flags);
    if (FAILED(hr))
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to present swap chain. IDXGISwapChain Present failed.");
#endif  
        CRC::MapCudaResource(cudaResources_[frameIndex_]);
        return hr;
    }

    frameIndex_ = (frameIndex_ + 1) % bufferCount_;

    CRC::MapCudaResource(cudaResources_[frameIndex_]);
    backBuffer_ = backSurfaces_[frameIndex_].get();

    return S_OK;
}

HRESULT CRCSwapChain::GetDesc(DXGI_SWAP_CHAIN_DESC *pDesc)
{
    return d3d11SwapChain_->GetDesc(pDesc);
}

std::unique_ptr<ICRCContainable> CRCIDXGISwapChainFactoryL0_0::Create(IDESC &desc) const
{
    CRC_SWAP_CHAIN_DESC* swapChainDesc = CRC::As<CRC_SWAP_CHAIN_DESC>(&desc);
    if (!swapChainDesc)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to create swap chain from desc. Desc is not CRC_SWAP_CHAIN_DESC.");
#endif
        return nullptr;
    }

    std::unique_ptr<CRCIDXGISwapChain> swapChain = std::make_unique<CRCIDXGISwapChain>
    (
        swapChainDesc->d3d11SwapChain_
    );

    return swapChain;
}

HRESULT CRCIDXGISwapChain::GetBuffer(UINT buffer, ICRCTexture2D*& texture)
{
    CRCID3D11Texture2D* backBuffer = CRC::As<CRCID3D11Texture2D>(texture);
    if (!backBuffer)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to get buffer. Texture is not CRCID3D11Texture2D.");
#endif
        return E_INVALIDARG;
    }

    d3d11SwapChain_->GetBuffer(buffer, __uuidof(ID3D11Texture2D), &backBuffer->Get());
}

HRESULT CRCIDXGISwapChain::ResizeBuffers
(
    UINT bufferCount, UINT width, UINT height, DXGI_FORMAT newFormat, UINT swapChainFlags
){
    return d3d11SwapChain_->ResizeBuffers(bufferCount, width, height, newFormat, swapChainFlags);
}

HRESULT CRCIDXGISwapChain::Present(UINT syncInterval, UINT flags)
{
    return d3d11SwapChain_->Present(syncInterval, flags);
}

HRESULT CRCIDXGISwapChain::GetDesc(DXGI_SWAP_CHAIN_DESC *pDesc)
{
    return d3d11SwapChain_->GetDesc(pDesc);
}
