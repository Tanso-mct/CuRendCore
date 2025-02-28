#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_swap_chain.cuh"

std::unique_ptr<ICRCContainable> CRCSwapChainFactoryL0_0::Create(IDESC &desc) const
{
    CRC_SWAP_CHAIN_DESC* swapChainDesc = CRC::As<CRC_SWAP_CHAIN_DESC>(&desc);
    if (!swapChainDesc) return nullptr;

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
    HRESULT hr = CRC::RegisterCudaResources
    (
        cudaResources_, cudaGraphicsRegisterFlagsSurfaceLoadStore,
        bufferCount_, d3d11SwapChain_.Get()
    );
    if (FAILED(hr)) throw std::runtime_error("Failed to create CRCSwapChain by registering CUDA resources.");

    backSurfaces_.resize(bufferCount_);
    for (int i = 0; i < bufferCount_; i++)
    {
        backSurfaces_[i] = CRC::CreateTexture2DFromCudaResource
        (
            cudaResources_[i], desc.BufferDesc.Width, desc.BufferDesc.Height, desc.BufferDesc.Format
        );

        if (!backSurfaces_[i])
        {
            throw std::runtime_error("Failed to create CRCSwapChain by creating CUDA surface objects.");
        }
    }

    hr = CRC::MapCudaResource(cudaResources_[frameIndex_]);
    if (FAILED(hr)) throw std::runtime_error("Failed to create CRCSwapChain by mapping CUDA resources.");

    backBuffer_ = backSurfaces_[frameIndex_].get();
}

CRCSwapChain::~CRCSwapChain()
{
    HRESULT hr = CRC::UnmapCudaResource(cudaResources_[frameIndex_]);
    if (FAILED(hr)) throw std::runtime_error("Failed to destroy CRCSwapChain by unmapping CUDA resources.");

    backSurfaces_.clear();

    CRC::UnregisterCudaResourcesAtSwapChain(cudaResources_, d3d11SwapChain_, frameIndex_, bufferCount_);
    if (FAILED(hr)) throw std::runtime_error("Failed to destroy CRCSwapChain by unregistering CUDA resources.");
    cudaResources_.clear();

    backBuffer_ = nullptr;
}

HRESULT CRCSwapChain::GetBuffer(UINT buffer, ICRCTexture2D*& texture)
{
    texture = backBuffer_;
    if (!texture) return E_INVALIDARG;
    return S_OK;
}

HRESULT CRCSwapChain::ResizeBuffers
(
    UINT bufferCount, UINT width, UINT height, DXGI_FORMAT newFormat, UINT swapChainFlags
){
    HRESULT hr = S_OK;
    hr = CRC::UnmapCudaResource(cudaResources_[frameIndex_]);
    if (FAILED(hr)) return hr;

    backSurfaces_.clear();

    hr = CRC::UnregisterCudaResourcesAtSwapChain(cudaResources_, d3d11SwapChain_, frameIndex_, bufferCount_);
    if (FAILED(hr)) return hr;
    cudaResources_.clear();

    hr = d3d11SwapChain_->ResizeBuffers(bufferCount, width, height, newFormat, swapChainFlags);
    if (FAILED(hr)) return hr;

    frameIndex_ = 0;

    hr = CRC::RegisterCudaResources
    (
        cudaResources_, cudaGraphicsRegisterFlagsSurfaceLoadStore, 
        bufferCount, d3d11SwapChain_.Get()
    );
    if (FAILED(hr)) return hr;

    backSurfaces_.resize(bufferCount);
    for (int i = 0; i < bufferCount; i++)
    {
        backSurfaces_[i] = CRC::CreateTexture2DFromCudaResource
        (
            cudaResources_[i], width, height, newFormat
        );

        if (!backSurfaces_[i]) return E_FAIL;
    }

    hr = CRC::MapCudaResource(cudaResources_[frameIndex_]);
    if (FAILED(hr)) return hr;

    backBuffer_ = backSurfaces_[frameIndex_].get();
    return S_OK;
}

HRESULT CRCSwapChain::Present(UINT syncInterval, UINT flags)
{
    HRESULT hr = CRC::UnmapCudaResource(cudaResources_[frameIndex_]);
    if (FAILED(hr)) return hr;

    hr = d3d11SwapChain_->Present(syncInterval, flags);
    if (FAILED(hr)) return hr;

    frameIndex_ = (frameIndex_ + 1) % bufferCount_;

    hr = CRC::MapCudaResource(cudaResources_[frameIndex_]);
    if (FAILED(hr)) return hr;

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
    if (!swapChainDesc) return nullptr;

    std::unique_ptr<CRCIDXGISwapChain> swapChain = std::make_unique<CRCIDXGISwapChain>
    (
        swapChainDesc->d3d11SwapChain_
    );

    return swapChain;
}

HRESULT CRCIDXGISwapChain::GetBuffer(UINT buffer, ICRCTexture2D*& texture)
{
    CRCID3D11Texture2D* backBuffer = CRC::As<CRCID3D11Texture2D>(texture);
    if (!backBuffer) return E_INVALIDARG;

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
