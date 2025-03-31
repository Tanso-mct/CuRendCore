#include "CuRendCore/include/pch.h"
#include "CuRendCore/include/funcs.cuh"

#include "CuRendCore/include/swap_chain.cuh"

std::unique_ptr<WACore::IContainable> CRCSwapChainFactoryL0_0::Create(IDESC &desc) const
{
    CRC_SWAP_CHAIN_DESC* swapChainDesc = WACore::As<CRC_SWAP_CHAIN_DESC>(&desc);
    if (!swapChainDesc)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to create swap chain from desc. Desc is not CRC_SWAP_CHAIN_DESC.");
#endif
    }

    std::unique_ptr<CRCSwapChain> swapChain = std::make_unique<CRCSwapChain>
    (
        swapChainDesc->d3d11Device_, swapChainDesc->d3d11SwapChain_, 
        swapChainDesc->GetDxgiDesc()
    );

    return swapChain;
}

CRCSwapChain::CRCSwapChain
(
    Microsoft::WRL::ComPtr<ID3D11Device>& d3d11Device,
    Microsoft::WRL::ComPtr<IDXGISwapChain> &d3d11SwapChain, 
    const DXGI_SWAP_CHAIN_DESC& desc
) 
: d3d11Device_(d3d11Device), d3d11SwapChain_(d3d11SwapChain)
, bufferCount_(desc.BufferCount), refreshRate_(desc.BufferDesc.RefreshRate)
{
    if (!(bufferCount_ == 2 || bufferCount_ == 3))
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to create swap chain. Buffer count is not 2 or 3.");
#endif
        return;
    }

    CRC::RegisterCudaResources
    (
        cudaResources_, cudaGraphicsRegisterFlagsSurfaceLoadStore,
        bufferCount_, d3d11SwapChain_.Get()
    );

    backSurfaces_.resize(bufferCount_);

    D3D11_TEXTURE2D_DESC backBufferDesc;
    ZeroMemory(&backBufferDesc, sizeof(D3D11_TEXTURE2D_DESC));
    backBufferDesc.Width = desc.BufferDesc.Width;
    backBufferDesc.Height = desc.BufferDesc.Height;
    backBufferDesc.MipLevels = 1;
    backBufferDesc.ArraySize = 1;
    backBufferDesc.Format = desc.BufferDesc.Format;
    backBufferDesc.SampleDesc.Count = 1;
    backBufferDesc.SampleDesc.Quality = 0;
    backBufferDesc.Usage = CRC::GetUsage(desc.BufferUsage);
    backBufferDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
    backBufferDesc.CPUAccessFlags = 0;
    backBufferDesc.MiscFlags = 0;

    for (int i = 0; i < bufferCount_; i++)
    {
        backSurfaces_[i] = CRC::CreatePtTexture2DFromCudaResource(cudaResources_[i], backBufferDesc);
    }

    CRC::MapCudaResource(cudaResources_[frameIndex_]);
    backBuffer_ = backSurfaces_[frameIndex_];

#ifndef NDEBUG
    CRC::Cout
    (
        "Swap chain created.", "\n",
        "Buffer count :", bufferCount_, "\n",
        "Refresh rate :", refreshRate_.Denominator, "/", refreshRate_.Numerator
    );
#endif
}

CRCSwapChain::~CRCSwapChain()
{
    CRC::UnmapCudaResource(cudaResources_[frameIndex_]);
    for (int i = 0; i < bufferCount_; i++)
    {
        delete backSurfaces_[i];
        backSurfaces_[i] = nullptr;
    }

    if (!presentExecuted_)
    {
        CRC::UnregisterSwapChainNotPresented
        (
            cudaResources_, d3d11Device_, d3d11SwapChain_, frameIndex_
        );
    }
    else
    {
        if (bufferCount_ == 2)
        {
            CRC::UnregisterSwapChain2Presented
            (
                cudaResources_, d3d11Device_, d3d11SwapChain_, frameIndex_
            );
        }
        else if (bufferCount_ == 3)
        {
            CRC::UnregisterSwapChain3Presented
            (
                cudaResources_, d3d11Device_, d3d11SwapChain_, frameIndex_
            );
        }
    }
    cudaResources_.clear();

    backBuffer_ = nullptr;

#ifndef NDEBUG
    CRC::Cout("Swap chain destroyed.");
#endif
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
    for (int i = 0; i < bufferCount_; i++)
    {
        delete backSurfaces_[i];
        backSurfaces_[i] = nullptr;
    }

    if (!presentExecuted_)
    {
        CRC::UnregisterSwapChainNotPresented
        (
            cudaResources_, d3d11Device_, d3d11SwapChain_, frameIndex_
        );
    }
    else
    {
        if (bufferCount_ == 2)
        {
            CRC::UnregisterSwapChain2Presented
            (
                cudaResources_, d3d11Device_, d3d11SwapChain_, frameIndex_
            );
        }
        else if (bufferCount_ == 3)
        {
            CRC::UnregisterSwapChain3Presented
            (
                cudaResources_, d3d11Device_, d3d11SwapChain_, frameIndex_
            );
        }
    }
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

    D3D11_TEXTURE2D_DESC backBufferDesc;
    ZeroMemory(&backBufferDesc, sizeof(D3D11_TEXTURE2D_DESC));
    backBufferDesc.Width = width;
    backBufferDesc.Height = height;
    backBufferDesc.MipLevels = 1;
    backBufferDesc.ArraySize = 1;
    backBufferDesc.Format = newFormat;
    backBufferDesc.SampleDesc.Count = 1;
    backBufferDesc.SampleDesc.Quality = 0;
    backBufferDesc.Usage = D3D11_USAGE_DEFAULT;
    backBufferDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
    backBufferDesc.CPUAccessFlags = 0;
    backBufferDesc.MiscFlags = 0;

    for (int i = 0; i < bufferCount; i++)
    {
        backSurfaces_[i] = CRC::CreatePtTexture2DFromCudaResource(cudaResources_[i], backBufferDesc);
    }

    CRC::MapCudaResource(cudaResources_[frameIndex_]);
    backBuffer_ = backSurfaces_[frameIndex_];

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
    if (!presentExecuted_) presentExecuted_ = true;

    CRC::MapCudaResource(cudaResources_[frameIndex_]);
    backBuffer_ = backSurfaces_[frameIndex_];

    return S_OK;
}

HRESULT CRCSwapChain::GetDesc(DXGI_SWAP_CHAIN_DESC *pDesc)
{
    return d3d11SwapChain_->GetDesc(pDesc);
}

std::unique_ptr<WACore::IContainable> CRCIDXGISwapChainFactoryL0_0::Create(IDESC &desc) const
{
    CRC_SWAP_CHAIN_DESC* swapChainDesc = WACore::As<CRC_SWAP_CHAIN_DESC>(&desc);
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

CRCIDXGISwapChain::CRCIDXGISwapChain(Microsoft::WRL::ComPtr<IDXGISwapChain>& d3d11SwapChain)
: d3d11SwapChain_(d3d11SwapChain)
{
    backBuffer_ = new CRCID3D11Texture2D();
}

CRCIDXGISwapChain::~CRCIDXGISwapChain()
{
    delete backBuffer_;
}

HRESULT CRCIDXGISwapChain::GetBuffer(UINT buffer, ICRCTexture2D *&texture)
{
    if (!d3d11SwapChain_)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to get buffer. IDXGISwapChain is nullptr.");
#endif
        return E_FAIL;
    }

    HRESULT hr = d3d11SwapChain_->GetBuffer
    (
        buffer, IID_PPV_ARGS(&WACore::As<CRCID3D11Texture2D>(backBuffer_)->Get())
    );
    if (FAILED(hr))
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to get buffer. IDXGISwapChain GetBuffer failed.");
#endif
    }

    texture = backBuffer_;

    return hr;
}

HRESULT CRCIDXGISwapChain::ResizeBuffers
(
    UINT bufferCount, UINT width, UINT height, DXGI_FORMAT newFormat, UINT swapChainFlags
){
    HRESULT hr = d3d11SwapChain_->ResizeBuffers(bufferCount, width, height, newFormat, swapChainFlags);
    if (FAILED(hr))
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to resize buffers. IDXGISwapChain ResizeBuffers failed.");
#endif
    }

    return hr;
}

HRESULT CRCIDXGISwapChain::Present(UINT syncInterval, UINT flags)
{
    HRESULT hr = d3d11SwapChain_->Present(syncInterval, flags);
    if (FAILED(hr))
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to present swap chain. IDXGISwapChain Present failed.");
#endif
    }

    return hr;
}

HRESULT CRCIDXGISwapChain::GetDesc(DXGI_SWAP_CHAIN_DESC *pDesc)
{
    return d3d11SwapChain_->GetDesc(pDesc);
}
