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

CRC_API D3D11_USAGE CRC::GetUsage(const DXGI_USAGE &usage)
{
    if (usage & DXGI_USAGE_READ_ONLY) 
    {
        return D3D11_USAGE_STAGING;
    }
    return D3D11_USAGE_DEFAULT;
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

CRC_API void CRC::GetCpuGpuRWFlags
(
    bool &cpuRead, bool &cpuWrite, bool &gpuRead, bool &gpuWrite, 
    const D3D11_USAGE &usage, const UINT &cpuAccessFlags
){
    switch (usage)
    {
    case D3D11_USAGE_DEFAULT:
        gpuRead = true;
        gpuWrite = true;
        break;

    case D3D11_USAGE_IMMUTABLE:
        gpuRead = true;
        break;

    case D3D11_USAGE_DYNAMIC:
        gpuRead = true;
        cpuWrite = true;
        break;

    case D3D11_USAGE_STAGING:
        gpuRead = true;
        gpuWrite = true;
        cpuRead = true;
        cpuWrite = true;
        break;

    case DXGI_USAGE_RENDER_TARGET_OUTPUT:
        gpuRead = true;
        gpuWrite = true;
        break;
    }

    switch (cpuAccessFlags)
    {
    case D3D11_CPU_ACCESS_READ:
        cpuRead = true;
        break;

    case D3D11_CPU_ACCESS_WRITE:
        cpuWrite = true;
        break;
    }
}

CRC_API bool CRC::NeedsWrite(const UINT &rcType)
{
    bool needsWrite = false;
    needsWrite = (rcType & (UINT)CRC_RESOURCE_TYPE::BUFFER_CPU_W) ? true : false;
    needsWrite = (rcType & (UINT)CRC_RESOURCE_TYPE::TEXTURE2D_CPU_W) ? true : false;

    return needsWrite;
}

CRC_API UINT CRC::GetCRCResourceType(const D3D11_BUFFER_DESC &desc)
{
    bool gpuR = false;
    bool gpuW = false;
    bool cpuR = false;
    bool cpuW = false;
    GetCpuGpuRWFlags(cpuR, cpuW, gpuR, gpuW, desc.Usage, desc.CPUAccessFlags);

    UINT type = 0;
    type |= cpuR ? (UINT)CRC_RESOURCE_TYPE::BUFFER_CPU_R : 0;
    type |= cpuW ? (UINT)CRC_RESOURCE_TYPE::BUFFER_CPU_W : 0;
    type |= gpuR ? (UINT)CRC_RESOURCE_TYPE::BUFFER_GPU_R : 0;
    type |= gpuW ? (UINT)CRC_RESOURCE_TYPE::BUFFER_GPU_W : 0;

    return type;
}

UINT CRC::GetCRCResourceType(const D3D11_TEXTURE2D_DESC &desc)
{
    bool gpuR = false;
    bool gpuW = false;
    bool cpuR = false;
    bool cpuW = false;
    GetCpuGpuRWFlags(cpuR, cpuW, gpuR, gpuW, desc.Usage, desc.CPUAccessFlags);

    UINT type = 0;
    type |= cpuR ? (UINT)CRC_RESOURCE_TYPE::TEXTURE2D_CPU_R : 0;
    type |= cpuW ? (UINT)CRC_RESOURCE_TYPE::TEXTURE2D_CPU_W : 0;
    type |= gpuR ? (UINT)CRC_RESOURCE_TYPE::TEXTURE2D_GPU_R : 0;
    type |= gpuW ? (UINT)CRC_RESOURCE_TYPE::TEXTURE2D_GPU_W : 0;

    return type;
}

CRC_API std::string CRC::GetCRCResourceTypeString(const UINT &rcType)
{
    std::string type = "";
    type += (rcType & (UINT)CRC_RESOURCE_TYPE::BUFFER_CPU_R) ? "CPU_R " : "";
    type += (rcType & (UINT)CRC_RESOURCE_TYPE::BUFFER_CPU_W) ? "CPU_W " : "";
    type += (rcType & (UINT)CRC_RESOURCE_TYPE::BUFFER_GPU_R) ? "GPU_R " : "";
    type += (rcType & (UINT)CRC_RESOURCE_TYPE::BUFFER_GPU_W) ? "GPU_W " : "";
    type += (rcType & (UINT)CRC_RESOURCE_TYPE::TEXTURE2D_CPU_R) ? "CPU_R " : "";
    type += (rcType & (UINT)CRC_RESOURCE_TYPE::TEXTURE2D_CPU_W) ? "CPU_W " : "";
    type += (rcType & (UINT)CRC_RESOURCE_TYPE::TEXTURE2D_GPU_R) ? "GPU_R " : "";
    type += (rcType & (UINT)CRC_RESOURCE_TYPE::TEXTURE2D_GPU_W) ? "GPU_W " : "";

    return type;
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
    cudaGraphicsResource_t& cudaResource, D3D11_TEXTURE2D_DESC& desc
){
    CRC::MapCudaResource(cudaResource);
    cudaArray_t backBufferArray = CRC::GetCudaMappedArray(cudaResource);

    std::unique_ptr<ICRCTexture2D> rtTexture = std::make_unique<CRCCudaResource>(desc);
    CRCCudaResource* backTexture = CRC::As<CRCCudaResource>(rtTexture.get());

    backTexture->Assign(backBufferArray, CRC::GetBytesPerPixel(desc.Format) * desc.Width * desc.Height);

    CRC::UnmapCudaResource(cudaResource);
    return rtTexture;
}
