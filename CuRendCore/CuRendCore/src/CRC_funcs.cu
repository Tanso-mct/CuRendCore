#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_window.h"
#include "CRC_scene.h"

#include "CRC_container.h"
#include "CRC_event.h"

#include "CRC_device.cuh"
#include "CRC_swap_chain.cuh"

HRESULT CRC::ShowWindowCRC(HWND& hWnd)
{
    if (!hWnd) return E_FAIL;

    ShowWindow(hWnd, SW_SHOW);
    UpdateWindow(hWnd);

    return S_OK;
}

CRC_API HRESULT CRC::CreateD3D11DeviceAndSwapChain
(
    const HWND& hWnd,
    Microsoft::WRL::ComPtr<ID3D11Device> &device, Microsoft::WRL::ComPtr<IDXGISwapChain> &swapChain
){
    // Setup swap chain
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

    UINT createDeviceFlags = 0;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0, };

    HRESULT hr = D3D11CreateDeviceAndSwapChain
    (
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags, featureLevelArray, 2, 
        D3D11_SDK_VERSION, &sd, &swapChain, &device, &featureLevel, nullptr
    );

    if (hr == DXGI_ERROR_UNSUPPORTED) // Try high-performance WARP software driver if hardware is not available.
    {
        hr = D3D11CreateDeviceAndSwapChain
        (
            nullptr, D3D_DRIVER_TYPE_WARP, nullptr, createDeviceFlags, featureLevelArray, 2, 
            D3D11_SDK_VERSION, &sd, &swapChain, &device, &featureLevel, nullptr
        );
    }

    return hr;
}

CRC_API HRESULT CRC::CreateCRCDeviceAndSwapChain
(
    Microsoft::WRL::ComPtr<ID3D11Device> &d3d11Device, Microsoft::WRL::ComPtr<IDXGISwapChain> &d3d11SwapChain, 
    std::unique_ptr<ICRCDevice> &crcDevice, std::unique_ptr<ICRCSwapChain> crcSwapChain
){
    return E_NOTIMPL;
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
        if (FAILED(hr)) return hr;

        cudaError_t err = cudaGraphicsD3D11RegisterResource
        (
            &cudaResources[i], buffers[i], flags
        );
        if (err != cudaSuccess) return E_FAIL;
    }

    for (int i = 0; i < bufferCountFromDesc; i++)
    {
        buffers[i]->Release();
    }

    return S_OK;
}

HRESULT CRC::RegisterCudaResource
(
    cudaGraphicsResource_t &cudaResource, const cudaGraphicsRegisterFlags &flags, 
    ID3D11Texture2D *d3d11Texture
){
    cudaError_t err = cudaGraphicsD3D11RegisterResource(&cudaResource, d3d11Texture, flags);
    if (err != cudaSuccess) return E_FAIL;

    return S_OK;
}

HRESULT CRC::UnregisterCudaResources(std::vector<cudaGraphicsResource_t> &cudaResources)
{
    for (int i = 0; i < cudaResources.size(); ++i) 
    {
        cudaError_t err = cudaGraphicsUnregisterResource(cudaResources[i]);
        if (err != cudaSuccess) return E_FAIL;
    }

    return S_OK;
}

HRESULT CRC::UnregisterCudaResource(cudaGraphicsResource_t &cudaResource)
{
    cudaError_t err = cudaGraphicsUnregisterResource(cudaResource);
    if (err != cudaSuccess) return E_FAIL;

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
        if (err != cudaSuccess) return E_FAIL;
    }

    d3d11SwapChain->Present(0, 0);
    cudaError_t err = cudaGraphicsUnregisterResource(cudaResources[frameIndex]);
    if (err != cudaSuccess) return E_FAIL;

    frameIndex = (frameIndex + 1) % bufferCount;

    return S_OK;
}

HRESULT CRC::MapCudaResource(cudaGraphicsResource_t& cudaResource, cudaStream_t stream)
{
    cudaError_t err = cudaGraphicsMapResources(1, &cudaResource, stream);
    if (err != cudaSuccess) return E_FAIL;

    return S_OK;
}

HRESULT CRC::UnmapCudaResource(cudaGraphicsResource_t& cudaResource, cudaStream_t stream)
{
    cudaError_t err = cudaGraphicsUnmapResources(1, &cudaResource, stream);
    if (err != cudaSuccess) return E_FAIL;

    return S_OK;
}

cudaArray_t CRC::GetCudaMappedArray(cudaGraphicsResource_t& cudaResource)
{
    cudaArray_t cudaArray;
    cudaError_t err = cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource, 0, 0);
    if (err != cudaSuccess) return nullptr;

    return cudaArray;
}
