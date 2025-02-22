#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_swap_chain.cuh"

std::unique_ptr<ICRCContainable> CRCSwapChainFactoryL0_0::Create(IDESC &desc) const
{
    CRC_SWAP_CHAIN_DESC* swapChainDesc = CRC::As<CRC_SWAP_CHAIN_DESC>(&desc);
    if (!swapChainDesc) return nullptr;

    std::unique_ptr<CRCSwapChain> swapChain = std::make_unique<CRCSwapChain>(swapChainDesc->GetD3D11SwapChain());
    return swapChain;
}

HRESULT CRCSwapChain::GetBuffer(UINT buffer, std::unique_ptr<ICRCContainable> &texture)
{
    return E_NOTIMPL;
}

HRESULT CRCSwapChain::ResizeBuffers(UINT bufferCount, UINT width, UINT height, DXGI_FORMAT newFormat, UINT swapChainFlags)
{
    return E_NOTIMPL;
}

HRESULT CRCSwapChain::Present(UINT syncInterval, UINT flags)
{
    return E_NOTIMPL;
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

HRESULT CRCIDXGISwapChain::GetBuffer(UINT buffer, std::unique_ptr<ICRCContainable> &texture)
{
    return E_NOTIMPL;
}

HRESULT CRCIDXGISwapChain::ResizeBuffers(UINT bufferCount, UINT width, UINT height, DXGI_FORMAT newFormat, UINT swapChainFlags)
{
    return E_NOTIMPL;
}

HRESULT CRCIDXGISwapChain::Present(UINT syncInterval, UINT flags)
{
    return E_NOTIMPL;
}
