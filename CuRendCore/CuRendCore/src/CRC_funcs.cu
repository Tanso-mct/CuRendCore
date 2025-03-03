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
    if (!hWnd)
    {
#ifndef NDEBUG
        CoutError("Failed to show window. Window handle is null.");
#endif
        throw std::runtime_error("Failed to show window. Window handle is null.");
    }

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

#ifndef NDEBUG
    if (FAILED(hr)) CoutError("Failed to create D3D11 device and swap chain.");
    else Cout("Created D3D11 device and swap chain.");
#endif

    return hr;
}

CRC_API HRESULT CRC::CreateCRCDeviceAndSwapChain
(
    CRC_DEVICE_DESC &deviceDesc, CRC_SWAP_CHAIN_DESC &swapChainDesc, 
    const ICRCFactory &deviceFactory, const ICRCFactory &swapChainFactory, 
    std::unique_ptr<ICRCContainable> &device, std::unique_ptr<ICRCContainable> &swapChain
){
    device = deviceFactory.Create(deviceDesc);
    if (!device)
    {
#ifndef NDEBUG
        CoutError("Failed to create CuRendCore device.");
#endif
        return E_FAIL;
    }

    swapChainDesc.d3d11SwapChain_->GetDesc(&swapChainDesc.GetDxgiDesc());

    swapChain = swapChainFactory.Create(swapChainDesc);
    if (!swapChain)
    {
#ifndef NDEBUG
        CoutError("Failed to create CuRendCore swap chain.");
#endif
        return E_FAIL;
    }

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
#ifndef NDEBUG
        CoutError("This DXGI_FORMAT is not supported by CuRendCore.");
#endif
        throw std::runtime_error("This DXGI_FORMAT is not supported by CuRendCore.");
    }
}

void CRC::CreateCudaChannelDescFromDXGIFormat(cudaChannelFormatDesc &channelDesc, const DXGI_FORMAT &format)
{
    switch (format)
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
        channelDesc = cudaCreateChannelDesc<uchar4>();
        break;

    default:
#ifndef NDEBUG
        CoutError("This DXGI_FORMAT is not supported by CuRendCore.");
#endif
        throw std::runtime_error("This DXGI_FORMAT is not supported by CuRendCore.");
    }
}

CRC_API void CRC::CheckCuda(cudaError_t call)
{
    if (call != cudaSuccess)
    {
        std::string code = std::to_string(call);
        std::string reason = cudaGetErrorString(call);
#ifndef NDEBUG
        CoutError("CUDA error occurred.", code, reason);
#endif
        throw std::runtime_error("CUDA error occurred. " + code + " " + reason);
    }
}

void CRC::RegisterCudaResources
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
            throw std::runtime_error("Failed to get buffers from DXGI swap chain.");
        }

        CRC::CheckCuda(cudaGraphicsD3D11RegisterResource
        (
            &cudaResources[i], buffers[i], flags
        ));
    }

    for (int i = 0; i < bufferCountFromDesc; i++)
    {
        buffers[i]->Release();
    }

#ifndef NDEBUG
    Cout("Registered CUDA resources.");
#endif
}

void CRC::RegisterCudaResource
(
    cudaGraphicsResource_t &cudaResource, const cudaGraphicsRegisterFlags &flags, 
    ID3D11Texture2D *d3d11Texture
){
    CRC::CheckCuda(cudaGraphicsD3D11RegisterResource(&cudaResource, d3d11Texture, flags));

#ifndef NDEBUG
    Cout("Registered CUDA resource.");
#endif
}

void CRC::UnregisterCudaResources(std::vector<cudaGraphicsResource_t> &cudaResources)
{
    for (int i = 0; i < cudaResources.size(); ++i) 
    {
        CRC::CheckCuda(cudaGraphicsUnregisterResource(cudaResources[i]));
    }

#ifndef NDEBUG
    Cout("Unregistered CUDA resources.");
#endif
}

void CRC::UnregisterCudaResource(cudaGraphicsResource_t &cudaResource)
{
    CRC::CheckCuda(cudaGraphicsUnregisterResource(cudaResource));

#ifndef NDEBUG
    CRC::Cout("Unregistered CUDA resource.");
#endif
}

void CRC::UnregisterCudaResourcesAtSwapChain
(
    std::vector<cudaGraphicsResource_t> &cudaResources, 
    Microsoft::WRL::ComPtr<IDXGISwapChain> &d3d11SwapChain, UINT &frameIndex, const UINT& bufferCount
){
    for (int i = 0; i < cudaResources.size(); ++i) 
    {
        if (i == frameIndex) continue;
        CRC::CheckCuda(cudaGraphicsUnregisterResource(cudaResources[i]));
    }

    HRESULT hr = d3d11SwapChain->Present(0, 0);
    if (FAILED(hr))
    {
#ifndef NDEBUG
        CRC::CoutError("Failed to present swap chain.");
#endif
        throw std::runtime_error("Failed to present swap chain.");
    }

    CRC::CheckCuda(cudaGraphicsUnregisterResource(cudaResources[frameIndex]));
    frameIndex = (frameIndex + 1) % bufferCount;

#ifndef NDEBUG
    Cout("Unregistered CUDA resources in swap chain.");
#endif
}

void CRC::MapCudaResource(cudaGraphicsResource_t& cudaResource, cudaStream_t stream)
{
    CRC::CheckCuda(cudaGraphicsMapResources(1, &cudaResource, stream));

#ifndef NDEBUG
    Cout("Mapped CUDA resource.");
#endif
}

void CRC::UnmapCudaResource(cudaGraphicsResource_t& cudaResource, cudaStream_t stream)
{
    CRC::CheckCuda(cudaGraphicsUnmapResources(1, &cudaResource, stream));

#ifndef NDEBUG
    Cout("Unmapped CUDA resource.");
#endif
}

cudaArray_t CRC::GetCudaMappedArray(cudaGraphicsResource_t& cudaResource)
{
    cudaArray_t cudaArray;
    CRC::CheckCuda(cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource, 0, 0));

    return cudaArray;
}

CRC_API std::unique_ptr<ICRCTexture2D> CRC::CreateTexture2DFromCudaResource
(
    cudaGraphicsResource_t& cudaResource, const UINT& width, const UINT& height, const DXGI_FORMAT& format
){
    CRC::MapCudaResource(cudaResource);
    cudaArray_t backBufferArray = CRC::GetCudaMappedArray(cudaResource);

    D3D11_TEXTURE2D_DESC desc;
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;

    std::unique_ptr<ICRCTexture2D> rtTexture = std::make_unique<CRCTexutre2DAttached>(desc);

    CRCTexutre2DAttached* backBuffer = CRC::As<CRCTexutre2DAttached>(rtTexture.get());
    if (!backBuffer)
    {
#ifndef NDEBUG
        CRC::CoutError("Failed to create texture2d from cuda resource by casting back surface to ICRCMem.");
#endif
        return nullptr;
    }

    backBuffer->Assign
    (
        backBufferArray, 
        CRC::GetBytesPerPixel(format) * width * height,
        width * CRC::GetBytesPerPixel(format)
    );

    CRC::UnmapCudaResource(cudaResource);
    return rtTexture;
}

CRC_API std::unique_ptr<ICRCTexture2D> CRC::CreateSurface2DFromCudaResource
(
    cudaGraphicsResource_t &cudaResource, const UINT &width, const UINT &height, const DXGI_FORMAT &format
){
    CRC::MapCudaResource(cudaResource);
    cudaArray_t backBufferArray = CRC::GetCudaMappedArray(cudaResource);

    D3D11_TEXTURE2D_DESC desc;
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;

    std::unique_ptr<ICRCTexture2D> rtTexture = std::make_unique<CRCSurface2D>(desc);

    CRCSurface2D* backBuffer = CRC::As<CRCSurface2D>(rtTexture.get());
    if (!backBuffer)
    {
#ifndef NDEBUG
        CRC::CoutError("Failed to create surface objects from cuda resource by casting back surface to ICRCMem.");
#endif
        throw std::runtime_error
        (
            "Failed to create surface objects from cuda resource by casting back surface to ICRCMem."
        );
    }

    backBuffer->Assign
    (
        backBufferArray, 
        CRC::GetBytesPerPixel(format) * width * height,
        width * CRC::GetBytesPerPixel(format)
    );

    CRC::UnmapCudaResource(cudaResource);
    return rtTexture;
}
