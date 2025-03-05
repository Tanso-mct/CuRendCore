#pragma once

#include "CRC_config.h"

#include <memory>
#include <Windows.h>
#include <utility>
#include <iostream>
#include <string_view>
#include <initializer_list>
#include <vector>
#include <string>

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

enum class CRC_RESOURCE_TYPE : UINT;

namespace CRC
{

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
CRC_API D3D11_USAGE GetUsage(const DXGI_USAGE& usage);

CRC_API void CreateCudaChannelDescFromDXGIFormat(cudaChannelFormatDesc& channelDesc, const DXGI_FORMAT& format);

CRC_API void GetCpuGpuRWFlags
(
    bool& cpuRead, bool& cpuWrite, bool& gpuRead, bool& gpuWrite, 
    const D3D11_USAGE& usage, const UINT& cpuAccessFlags
);

CRC_API bool NeedsWrite(const UINT& rcType);

CRC_API UINT GetCRCResourceType(const D3D11_BUFFER_DESC& desc);
CRC_API UINT GetCRCResourceType(const D3D11_TEXTURE2D_DESC& desc);

CRC_API std::string GetCRCResourceTypeString(const UINT& rcType);

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

CRC_API void RegisterCudaResources
(
    std::vector<cudaGraphicsResource_t>& cudaResources, const cudaGraphicsRegisterFlags& flags,
    const UINT& bufferCount, IDXGISwapChain* d3d11SwapChain
);
CRC_API void RegisterCudaResource
(
    cudaGraphicsResource_t& cudaResource, const cudaGraphicsRegisterFlags& flags,
    ID3D11Texture2D* d3d11Texture
);

CRC_API void UnregisterCudaResources(std::vector<cudaGraphicsResource_t>& cudaResources);
CRC_API void UnregisterCudaResource(cudaGraphicsResource_t& cudaResource);

/**
 * @brief Un registers all CUDA resources from the swap chain.
 * At this time, if SwapChain has been presented at least once, unregistering the buffer 
 * that will become the next display buffer directly will cause windows to freeze, 
 * so unregister after presenting and shifting the buffer.
 * The error is probably due to the fact that it is tied to RenderTarget, etc.
 */
CRC_API void UnregisterCudaResourcesAtSwapChain
(
    std::vector<cudaGraphicsResource_t>& cudaResources, 
    Microsoft::WRL::ComPtr<IDXGISwapChain>& d3d11SwapChain, UINT& frameIndex, const UINT& bufferCount
);

CRC_API void MapCudaResource(cudaGraphicsResource_t& cudaResource, cudaStream_t stream = 0);
CRC_API void UnmapCudaResource(cudaGraphicsResource_t& cudaResource, cudaStream_t stream = 0);
cudaArray_t GetCudaMappedArray(cudaGraphicsResource_t& cudaResource);

CRC_API std::unique_ptr<ICRCTexture2D> CreateTexture2DFromCudaResource
(
    cudaGraphicsResource_t& cudaResource, D3D11_TEXTURE2D_DESC& desc
);

}