#pragma once

#include "CRC_config.h"

#include <memory>
#include <Windows.h>
#include <utility>
#include <iostream>
#include <string_view>
#include <initializer_list>
#include <vector>

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CRCCore;
class CRC_WINDOW_DESC; 
struct CRC_SCENE_DESC;

class ICRCContainable;
class ICRCContainer;

class ICRCWinMsgEvent;

class ICRCFactory;

class ICRCDevice;
class CRC_DEVICE_DESC;

class ICRCSwapChain;
class CRC_SWAP_CHAIN_DESC;

class ICRCTexture2D;

namespace CRC
{

template <typename T, typename S>
T* As(S* source)
{
    T* target = dynamic_cast<T*>(source);
    return target;
}

template <typename T, typename S>
std::unique_ptr<T> UniqueAs(std::unique_ptr<S>& source)
{
    T* target = dynamic_cast<T*>(source.get());

    if (target) return std::unique_ptr<T>(static_cast<T*>(source.release()));
    else return nullptr;
}

CRC_API HRESULT ShowWindowCRC(HWND& hWnd);

CRC_API HRESULT CreateD3D11DeviceAndSwapChain
(
    CRC_SWAP_CHAIN_DESC& desc,
    Microsoft::WRL::ComPtr<ID3D11Device>& device, Microsoft::WRL::ComPtr<IDXGISwapChain>& swapChain
);

CRC_API HRESULT CreateCRCDeviceAndSwapChain
(
    CRC_DEVICE_DESC& deviceDesc, CRC_SWAP_CHAIN_DESC& swapChainDesc,
    const ICRCFactory& deviceFactory, const ICRCFactory& swapChainFactory,
    std::unique_ptr<ICRCContainable>& device, std::unique_ptr<ICRCContainable>& swapChain
);

CRC_API UINT GetBytesPerPixel(const DXGI_FORMAT& format);
HRESULT CreateCudaChannelDescFromDXGIFormat(cudaChannelFormatDesc& channelDesc, const DXGI_FORMAT& format);

struct PairHash 
{
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const 
    {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

struct PairEqual 
{
    template <typename T1, typename T2>
    bool operator()(const std::pair<T1, T2>& lhs, const std::pair<T1, T2>& rhs) const 
    {
        return lhs == rhs;
    }
};

CRC_API void CheckCuda(cudaError_t call);

template <typename... Args>
CRC_API void Cout(Args&... args)
{
    std::cout << CRC::C_COLOR_MSG << CRC::C_TAG << CRC::C_COLOR_RESET << " ";
    std::initializer_list<int> ilist = {(std::cout << args << " ", 0)...};

    if (ilist.size() == 1) std::cout << std::endl;
    else std::cout << std::endl << CRC::C_COLOR_MSG << CRC::C_TAG_END << CRC::C_COLOR_RESET << std::endl;
}

template <typename... Args>
CRC_API void CoutError(Args&... args)
{
    std::cout << CRC::C_COLOR_ERROR << CRC::C_TAG << CRC::C_COLOR_RESET << " ";
    std::initializer_list<int> ilist = {(std::cout << args << " ", 0)...};
    
    if (ilist.size() == 1) std::cout << std::endl;
    else std::cout << std::endl << CRC::C_COLOR_ERROR << CRC::C_TAG_END << CRC::C_COLOR_RESET << std::endl;
}

template <typename... Args>
CRC_API void CoutWarning(Args&... args)
{
    std::cout << CRC::C_COLOR_WARNING << CRC::C_TAG << CRC::C_COLOR_RESET << " ";
    std::initializer_list<int> ilist = {(std::cout << args << " ", 0)...};
    
    if (ilist.size() == 1) std::cout << std::endl;
    else std::cout << std::endl << CRC::C_COLOR_WARNING << CRC::C_TAG_END << CRC::C_COLOR_RESET << std::endl;
}

HRESULT RegisterCudaResources
(
    std::vector<cudaGraphicsResource_t>& cudaResources, const cudaGraphicsRegisterFlags& flags,
    const UINT& bufferCount, IDXGISwapChain* d3d11SwapChain
);
HRESULT RegisterCudaResource
(
    cudaGraphicsResource_t& cudaResource, const cudaGraphicsRegisterFlags& flags,
    ID3D11Texture2D* d3d11Texture
);

HRESULT UnregisterCudaResources(std::vector<cudaGraphicsResource_t>& cudaResources);
HRESULT UnregisterCudaResource(cudaGraphicsResource_t& cudaResource);

/**
 * @brief Un registers all CUDA resources from the swap chain.
 * At this time, if SwapChain has been presented at least once, unregistering the buffer 
 * that will become the next display buffer directly will cause windows to freeze, 
 * so unregister after presenting and shifting the buffer.
 * The error is probably due to the fact that it is tied to RenderTarget, etc.
 */
HRESULT UnregisterCudaResourcesAtSwapChain
(
    std::vector<cudaGraphicsResource_t>& cudaResources, 
    Microsoft::WRL::ComPtr<IDXGISwapChain>& d3d11SwapChain, UINT& frameIndex, const UINT& bufferCount
);

HRESULT MapCudaResource(cudaGraphicsResource_t& cudaResource, cudaStream_t stream = 0);
HRESULT UnmapCudaResource(cudaGraphicsResource_t& cudaResource, cudaStream_t stream = 0);
cudaArray_t GetCudaMappedArray(cudaGraphicsResource_t& cudaResource);

CRC_API std::unique_ptr<ICRCTexture2D> CreateTexture2DFromCudaResource
(
    cudaGraphicsResource_t& cudaResource, const UINT& width, const UINT& height, const DXGI_FORMAT& format
);

CRC_API std::unique_ptr<ICRCTexture2D> CreateSurface2DFromCudaResource
(
    cudaGraphicsResource_t& cudaResource, const UINT& width, const UINT& height, const DXGI_FORMAT& format
);

}