#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_window.h"
#include "CRC_scene.h"

#include "CRC_container.h"
#include "CRC_event.h"

#include "CRC_device.cuh"
#include "CRC_swap_chain.cuh"

#include "CRC_texture.cuh"

HRESULT CRC::ShowWindowCRC(HWND& hWnd)
{
    if (!hWnd) return E_FAIL;

    ShowWindow(hWnd, SW_SHOW);
    UpdateWindow(hWnd);

    return S_OK;
}

CRC_API HRESULT CRC::CreateD3D11DeviceAndSwapChain
(
    CRC_SWAP_CHAIN_DESC& desc,
    Microsoft::WRL::ComPtr<ID3D11Device> &device, Microsoft::WRL::ComPtr<IDXGISwapChain> &swapChain
){
    UINT createDeviceFlags = 0;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0, };

    HRESULT hr = D3D11CreateDeviceAndSwapChain
    (
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags, featureLevelArray, 2, 
        D3D11_SDK_VERSION, &desc.GetDxgiDesc(), &swapChain, &device, &featureLevel, nullptr
    );

    if (hr == DXGI_ERROR_UNSUPPORTED) // Try high-performance WARP software driver if hardware is not available.
    {
        hr = D3D11CreateDeviceAndSwapChain
        (
            nullptr, D3D_DRIVER_TYPE_WARP, nullptr, createDeviceFlags, featureLevelArray, 2, 
            D3D11_SDK_VERSION, &desc.GetDxgiDesc(), &swapChain, &device, &featureLevel, nullptr
        );
    }

    return hr;
}

CRC_API HRESULT CRC::CreateCRCDeviceAndSwapChain
(
    CRC_DEVICE_DESC &deviceDesc, CRC_SWAP_CHAIN_DESC &swapChainDesc, 
    const ICRCFactory &deviceFactory, const ICRCFactory &swapChainFactory, 
    std::unique_ptr<ICRCContainable> &device, std::unique_ptr<ICRCContainable> &swapChain
){
    device = deviceFactory.Create(deviceDesc);
    if (!device) return E_FAIL;

    swapChain = swapChainFactory.Create(swapChainDesc);
    if (!swapChain) return E_FAIL;

    return S_OK;
}

UINT CRC::GetBytesPerPixel(const DXGI_FORMAT &format)
{
    switch (format)
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
        return 4;

    default:
        throw std::runtime_error("This DXGI_FORMAT is not supported by CuRendCore.");
        return 0;
    }
}

HRESULT CRC::CreateCudaChannelDescFromDXGIFormat(cudaChannelFormatDesc &channelDesc, const DXGI_FORMAT &format)
{
    switch (format)
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
        channelDesc = cudaCreateChannelDesc<uchar4>();
        return S_OK;

    default:
        throw std::runtime_error("This DXGI_FORMAT is not supported by CuRendCore.");
        return E_FAIL;
    }
}

CRC_API void CRC::CheckCuda(cudaError_t call)
{
    if (call != cudaSuccess)
    {
        std::string err = "[CUDA ERROR] Code: " + std::to_string(call) + ", Reason: " + cudaGetErrorString(call);
#ifndef NDEBUG
        CoutError(err);
#endif
        throw std::runtime_error(err);
    }
}

HRESULT CRC::RegisterCudaResources
(
    std::vector<cudaGraphicsResource_t> &cudaResources, const cudaGraphicsRegisterFlags &flags, 
    const UINT &bufferCount, IDXGISwapChain *d3d11SwapChain
){
    cudaResources.resize(bufferCount);
    std::vector<ID3D11Texture2D*> buffers(bufferCount);

    DXGI_SWAP_CHAIN_DESC desc;
    d3d11SwapChain->GetDesc(&desc);
    UINT bufferCountFromDesc = desc.BufferCount;

    for (UINT i = 0; i < bufferCountFromDesc; i++)
    {
        HRESULT hr = d3d11SwapChain->GetBuffer(i, __uuidof(ID3D11Texture2D), (void**)&buffers[i]);
        if (FAILED(hr))
        {
#ifndef NDEBUG
            CoutError("Failed to get buffers from DXGI swap chain.");
#endif
            return E_FAIL;
        }

        cudaError_t err = cudaGraphicsD3D11RegisterResource
        (
            &cudaResources[i], buffers[i], flags
        );
        if (err != cudaSuccess)
        {
#ifndef NDEBUG
            CoutError("Failed to register CUDA resources.");
#endif
            return E_FAIL;
        }
    }

    for (int i = 0; i < bufferCountFromDesc; i++)
    {
        buffers[i]->Release();
    }

#ifndef NDEBUG
    Cout("Registered CUDA resources.");
#endif

    return S_OK;
}

HRESULT CRC::RegisterCudaResource
(
    cudaGraphicsResource_t &cudaResource, const cudaGraphicsRegisterFlags &flags, 
    ID3D11Texture2D *d3d11Texture
){
    cudaError_t err = cudaGraphicsD3D11RegisterResource(&cudaResource, d3d11Texture, flags);
    if (err != cudaSuccess)
    {
#ifndef NDEBUG
        CoutError("Failed to register CUDA resource.");
#endif
        return E_FAIL;
    }

#ifndef NDEBUG
    Cout("Registered CUDA resource.");
#endif

    return S_OK;
}

HRESULT CRC::UnregisterCudaResources(std::vector<cudaGraphicsResource_t> &cudaResources)
{
    for (int i = 0; i < cudaResources.size(); ++i) 
    {
        cudaError_t err = cudaGraphicsUnregisterResource(cudaResources[i]);
        if (err != cudaSuccess)
        {
#ifndef NDEBUG
            CoutError("Failed to unregister CUDA resources.");
#endif
            return E_FAIL;
        }
    }

#ifndef NDEBUG
    Cout("Unregistered CUDA resources.");
#endif

    return S_OK;
}

HRESULT CRC::UnregisterCudaResource(cudaGraphicsResource_t &cudaResource)
{
    cudaError_t err = cudaGraphicsUnregisterResource(cudaResource);
    if (err != cudaSuccess)
    {
#ifndef NDEBUG
        CoutError("Failed to unregister CUDA resource.");
#endif
        return E_FAIL;
    }

#ifndef NDEBUG
    Cout("Unregistered CUDA resource.");
#endif

    return S_OK;
}

HRESULT CRC::UnregisterCudaResourcesAtSwapChain
(
    std::vector<cudaGraphicsResource_t> &cudaResources, 
    Microsoft::WRL::ComPtr<IDXGISwapChain> &d3d11SwapChain, UINT &frameIndex, const UINT& bufferCount
){
    for (int i = 0; i < cudaResources.size(); ++i) 
    {
        if (i == frameIndex) continue;

        cudaError_t err = cudaGraphicsUnregisterResource(cudaResources[i]);
        if (err != cudaSuccess)
        {
#ifndef NDEBUG
            CoutError("Failed to unregister CUDA resources in swap chain.");
#endif
            return E_FAIL;
        }
    }

    d3d11SwapChain->Present(0, 0);
    cudaError_t err = cudaGraphicsUnregisterResource(cudaResources[frameIndex]);
    if (err != cudaSuccess)
    {
#ifndef NDEBUG
        CoutError("Failed to unregister CUDA resources in swap chain.");
#endif
        return E_FAIL;
    }

    frameIndex = (frameIndex + 1) % bufferCount;

#ifndef NDEBUG
    Cout("Unregistered CUDA resources in swap chain.");
#endif

    return S_OK;
}

HRESULT CRC::MapCudaResource(cudaGraphicsResource_t& cudaResource, cudaStream_t stream)
{
    cudaError_t err = cudaGraphicsMapResources(1, &cudaResource, stream);
    if (err != cudaSuccess)
    {
#ifndef NDEBUG
        CoutError("Failed to map CUDA resource.");
#endif
        return E_FAIL;
    }

#ifndef NDEBUG
    Cout("Mapped CUDA resource.");
#endif

    return S_OK;
}

HRESULT CRC::UnmapCudaResource(cudaGraphicsResource_t& cudaResource, cudaStream_t stream)
{
    cudaError_t err = cudaGraphicsUnmapResources(1, &cudaResource, stream);
    if (err != cudaSuccess)
    {
#ifndef NDEBUG
        CoutError("Failed to unmap CUDA resource.");
#endif
        return E_FAIL;
    }

#ifndef NDEBUG
    Cout("Unmapped CUDA resource.");
#endif

    return S_OK;
}

cudaArray_t CRC::GetCudaMappedArray(cudaGraphicsResource_t& cudaResource)
{
    cudaArray_t cudaArray;
    cudaError_t err = cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource, 0, 0);
    if (err != cudaSuccess)
    {
#ifndef NDEBUG
        CoutError("Failed to get CUDA mapped array.");
#endif
        return nullptr;
    }

    return cudaArray;
}

CRC_API std::unique_ptr<ICRCTexture2D> CRC::CreateCudaSurfaceObjects
(
    cudaGraphicsResource_t& cudaResource, const UINT& width, const UINT& height, const DXGI_FORMAT& format
){
    HRESULT hr = CRC::MapCudaResource(cudaResource);
    if (FAILED(hr))
    {
#ifndef NDEBUG
        CRC::CoutError("Failed to create surface objects by mapping CUDA resources.");
#endif
        return nullptr;
    }

    cudaArray_t backBufferArray = CRC::GetCudaMappedArray(cudaResource);
    if (!backBufferArray)
    {
#ifndef NDEBUG
        CRC::CoutError("Failed to create surface objects by getting CUDA mapped array.");
#endif
        return nullptr;
    }

    std::unique_ptr<ICRCTexture2D> rtTexture = std::make_unique<CRCTextureSurface>
    (
        CRC::GetBytesPerPixel(format) * width * height, 
        CRC::GetBytesPerPixel(format) * width, 
        CRC::GetBytesPerPixel(format) * width * height
    );

    CRCTextureSurface* backSurface = CRC::As<CRCTextureSurface>(rtTexture.get());
    if (!backSurface)
    {
#ifndef NDEBUG
        CRC::CoutError("Failed to create surface objects by casting back surface to ICRCMem.");
#endif
        return nullptr;
    }

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = backBufferArray;
    hr = CRC::CreateCudaSurfaceObject(backSurface->GetSurfaceObj(), resDesc);

    hr = CRC::UnmapCudaResource(cudaResource);
    if (FAILED(hr))
    {
#ifndef NDEBUG
        CRC::CoutError("Failed to create surface objects by unmapping CUDA resources.");
#endif
        return nullptr;
    }

    return rtTexture;
}

HRESULT CRC::CreateCudaSurfaceObject(cudaSurfaceObject_t &surfaceObject, const cudaResourceDesc &desc)
{
    cudaError_t err = cudaCreateSurfaceObject(&surfaceObject, &desc);
    if (err != cudaSuccess)
    {
#ifndef NDEBUG
        CoutError("Failed to create CUDA surface object.");
#endif
        return E_FAIL;
    }

#ifndef NDEBUG
    Cout("Created CUDA surface object.");
#endif

    return S_OK;
}
