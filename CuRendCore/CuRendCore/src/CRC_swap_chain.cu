#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_swap_chain.cuh"

std::unique_ptr<ICRCContainable> CRCSwapChainFactoryL0_0::Create(IDESC &desc) const
{
    CRC_SWAP_CHAIN_DESC* swapChainDesc = CRC::As<CRC_SWAP_CHAIN_DESC>(&desc);
    if (!swapChainDesc) return nullptr;

    std::unique_ptr<CRCSwapChain> swapChain = std::make_unique<CRCSwapChain>
    (
        swapChainDesc->GetD3D11SwapChain(),
        swapChainDesc->BufferCount(),
        swapChainDesc->BufferUsage(),
        swapChainDesc->RefreshRate(),
        swapChainDesc->SwapEffect()
    );

    return swapChain;
}

CRCSwapChain::CRCSwapChain
(
    Microsoft::WRL::ComPtr<IDXGISwapChain> &d3d11SwapChain, 
    UINT bufferCount, DXGI_USAGE bufferUsage, DXGI_RATIONAL refreshRate, DXGI_SWAP_EFFECT swapEffect
) 
: d3d11SwapChain_(d3d11SwapChain)
, bufferCount_(bufferCount), bufferUsage_(bufferUsage), refreshRate_(refreshRate), swapEffect_(swapEffect)
{
    HRESULT hr = CRC::RegisterCudaResources
    (
        cudaResources_, cudaGraphicsRegisterFlagsNone,
        bufferCount, d3d11SwapChain_.Get()
    );
    if (FAILED(hr))
    {
#ifndef NDEBUG
        CRC::CoutError("Failed to create CRCSwapChain by registering CUDA resources.");
#endif
        throw std::runtime_error("Failed to create CRCSwapChain by registering CUDA resources.");
    }

    hr = CRC::MapCudaResource(cudaResources_[frameIndex_]);
    if (FAILED(hr))
    {
#ifndef NDEBUG
        CRC::CoutError("Failed to create CRCSwapChain by mapping CUDA resources.");
#endif
        throw std::runtime_error("Failed to create CRCSwapChain by mapping CUDA resources.");
    }

    cudaArray_t backBufferArray = CRC::GetCudaMappedArray(cudaResources_[frameIndex_]);
    if (!backBufferArray)
    {
#ifndef NDEBUG
        CRC::CoutError("Failed to create CRCSwapChain by getting CUDA mapped array.");
#endif
        throw std::runtime_error("Failed to create CRCSwapChain by getting CUDA mapped array.");
    }

    backBuffer_ = std::make_unique<CRCTexture2D>();
    backBuffer_->GetPtr() = (void*)backBufferArray;
}

CRCSwapChain::~CRCSwapChain()
{
    CRC::UnmapCudaResource(cudaResources_[frameIndex_]);
    CRC::UnregisterCudaResourcesAtSwapChain(cudaResources_, d3d11SwapChain_, frameIndex_, bufferCount_);

    backBuffer_->GetPtr() = nullptr;
    backBuffer_.reset();
}

HRESULT CRCSwapChain::GetBuffer(UINT buffer, ICRCTexture2D *&texture)
{
    texture = backBuffer_.get();
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

    hr = CRC::UnregisterCudaResourcesAtSwapChain(cudaResources_, d3d11SwapChain_, frameIndex_, bufferCount_);
    if (FAILED(hr)) return hr;

    hr = d3d11SwapChain_->ResizeBuffers(bufferCount, width, height, newFormat, swapChainFlags);
    if (FAILED(hr)) return hr;

    frameIndex_ = 0;

    hr = CRC::RegisterCudaResources
    (
        cudaResources_, cudaGraphicsRegisterFlagsNone, 
        bufferCount, d3d11SwapChain_.Get()
    );
    if (FAILED(hr)) return hr;

    hr = CRC::MapCudaResource(cudaResources_[frameIndex_]);
    if (FAILED(hr)) return hr;

    cudaArray_t backBufferArray = CRC::GetCudaMappedArray(cudaResources_[frameIndex_]);
    if (!backBufferArray) return E_FAIL;

    backBuffer_->GetPtr() = (void*)backBufferArray;

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

    cudaArray_t backBufferArray = CRC::GetCudaMappedArray(cudaResources_[frameIndex_]);
    if (!backBufferArray) return E_FAIL;

    backBuffer_->GetPtr() = (void*)backBufferArray;

    return S_OK;
}

std::unique_ptr<ICRCContainable> CRCIDXGISwapChainFactoryL0_0::Create(IDESC &desc) const
{
    CRC_SWAP_CHAIN_DESC* swapChainDesc = CRC::As<CRC_SWAP_CHAIN_DESC>(&desc);
    if (!swapChainDesc) return nullptr;

    std::unique_ptr<CRCIDXGISwapChain> swapChain = std::make_unique<CRCIDXGISwapChain>
    (
        swapChainDesc->GetD3D11SwapChain()
    );

    return swapChain;
}

HRESULT CRCIDXGISwapChain::GetBuffer(UINT buffer, ICRCTexture2D *&texture)
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
