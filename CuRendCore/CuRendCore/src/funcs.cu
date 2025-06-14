﻿#include "CuRendCore/include/pch.h"
#include "CuRendCore/include/funcs.cuh"

#include "CuRendCore/include/window.h"
#include "CuRendCore/include/scene.h"

#include "CuRendCore/include/device.cuh"
#include "CuRendCore/include/swap_chain.cuh"

#include "CuRendCore/include/texture.cuh"

HRESULT CRC::ShowWindowCRC(HWND& hWnd)
{
    if (!hWnd)
    {
#ifndef NDEBUG
        CRC::CoutDebug({"Failed to show window. Window handle is null."});
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

    if (FAILED(hr))
    {
#ifndef NDEBUG
        CRC::CoutDebug({"Failed to create D3D11 device and swap chain."});
#endif
        throw std::runtime_error("Failed to create D3D11 device and swap chain.");
    }

#ifndef NDEBUG
    CRC::CoutDebug({"Created D3D11 device and swap chain."});
#endif

    return hr;
}

CRC_API HRESULT CRC::CreateCRCDeviceAndSwapChain
(
    CRC_DEVICE_DESC &deviceDesc, CRC_SWAP_CHAIN_DESC &swapChainDesc, 
    const ICRCFactory &deviceFactory, const ICRCFactory &swapChainFactory, 
    std::unique_ptr<WACore::IContainable> &device, std::unique_ptr<WACore::IContainable> &swapChain
){
    device = deviceFactory.Create(deviceDesc);
    if (!device)
    {
#ifndef NDEBUG
        CRC::CoutDebug({"Failed to create CuRendCore device."});
#endif
        return E_FAIL;
    }

    swapChainDesc.d3d11SwapChain_->GetDesc(&swapChainDesc.GetDxgiDesc());

    swapChain = swapChainFactory.Create(swapChainDesc);
    if (!swapChain)
    {
#ifndef NDEBUG
        CRC::CoutDebug({"Failed to create CuRendCore swap chain."});
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
        CRC::CoutDebug({"This DXGI_FORMAT is not supported by CuRendCore."});
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
        CRC::CoutDebug({"This DXGI_FORMAT is not supported by CuRendCore."});
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
    needsWrite = (rcType & (UINT)CRC_RESOURCE_TYPE::CPU_W) ? true : false;

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
    type |= cpuR ? (UINT)CRC_RESOURCE_TYPE::CPU_R : 0;
    type |= cpuW ? (UINT)CRC_RESOURCE_TYPE::CPU_W : 0;
    type |= gpuR ? (UINT)CRC_RESOURCE_TYPE::GPU_R : 0;
    type |= gpuW ? (UINT)CRC_RESOURCE_TYPE::GPU_W : 0;

    type |= (UINT)CRC_RESOURCE_TYPE::BUFFER;

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
    type |= cpuR ? (UINT)CRC_RESOURCE_TYPE::CPU_R : 0;
    type |= cpuW ? (UINT)CRC_RESOURCE_TYPE::CPU_W : 0;
    type |= gpuR ? (UINT)CRC_RESOURCE_TYPE::GPU_R : 0;
    type |= gpuW ? (UINT)CRC_RESOURCE_TYPE::GPU_W : 0;

    type |= (UINT)CRC_RESOURCE_TYPE::TEXTURE2D;

    return type;
}

CRC_API std::string CRC::GetCRCResourceTypeString(const UINT &rcType)
{
    std::string type = "";
    type += (rcType & (UINT)CRC_RESOURCE_TYPE::UNKNOWN) ? "UNKNOWN " : "";
    type += (rcType & (UINT)CRC_RESOURCE_TYPE::CRC_RESOURCE) ? "CRC_RESOURCE " : "";
    type += (rcType & (UINT)CRC_RESOURCE_TYPE::D3D11_RESOURCE) ? "D3D11_RESOURCE " : "";

    type += (rcType & (UINT)CRC_RESOURCE_TYPE::BUFFER) ? "BUFFER " : "";
    type += (rcType & (UINT)CRC_RESOURCE_TYPE::TEXTURE2D) ? "TEXTURE2D " : "";

    type += (rcType & (UINT)CRC_RESOURCE_TYPE::CPU_R) ? "CPU_R " : "";
    type += (rcType & (UINT)CRC_RESOURCE_TYPE::CPU_W) ? "CPU_W " : "";
    type += (rcType & (UINT)CRC_RESOURCE_TYPE::GPU_R) ? "GPU_R " : "";
    type += (rcType & (UINT)CRC_RESOURCE_TYPE::GPU_W) ? "GPU_W " : "";

    return type;
}

CRC_API void CRC::CheckCuda(cudaError_t call)
{
    if (call != cudaSuccess)
    {
        std::string code = std::to_string(call);
        std::string reason = cudaGetErrorString(call);
#ifndef NDEBUG
        CRC::CoutDebug({"CUDA error occurred.", code, reason});
#endif
        throw std::runtime_error("CUDA error occurred. " + code + " " + reason);
    }
}

CRC_API std::unique_ptr<WACore::ConsoleOuter> &CRC::GetConsoleOuter()
{
    static std::unique_ptr<WACore::ConsoleOuter> consoleOuter = std::make_unique<WACore::ConsoleOuter>();
    consoleOuter->startTag_ = "[CuRendCore]";
    return consoleOuter;
}

CRC_API void CRC::Cout(std::initializer_list<std::string_view> args)
{
    GetConsoleOuter()->Cout(args);
}

CRC_API void CRC::CoutErr(std::initializer_list<std::string_view> args)
{
    GetConsoleOuter()->CoutErr(args);
}

CRC_API void CRC::CoutWrn(std::initializer_list<std::string_view> args)
{
    GetConsoleOuter()->CoutWrn(args);
}

CRC_API void CRC::CoutInfo(std::initializer_list<std::string_view> args)
{
    GetConsoleOuter()->CoutInfo(args);
}

CRC_API void CRC::CoutDebug(std::initializer_list<std::string_view> args)
{
    GetConsoleOuter()->CoutDebug(args);
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
            CRC::CoutDebug({"Failed to get buffers from DXGI swap chain."});
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
    CRC::CoutDebug({"Registered CUDA resources."});
#endif
}

void CRC::RegisterCudaResource
(
    cudaGraphicsResource_t &cudaResource, const cudaGraphicsRegisterFlags &flags, 
    ID3D11Texture2D *d3d11Texture
){
    CRC::CheckCuda(cudaGraphicsD3D11RegisterResource(&cudaResource, d3d11Texture, flags));

#ifndef NDEBUG
    CRC::CoutDebug({"Registered CUDA resource."});
#endif
}

void CRC::UnregisterCudaResources(std::vector<cudaGraphicsResource_t> &cudaResources)
{
    for (int i = 0; i < cudaResources.size(); ++i) 
    {
        CRC::CheckCuda(cudaGraphicsUnregisterResource(cudaResources[i]));
    }

#ifndef NDEBUG
    CRC::CoutDebug({"Unregistered CUDA resources."});
#endif
}

void CRC::UnregisterCudaResource(cudaGraphicsResource_t &cudaResource)
{
    CRC::CheckCuda(cudaGraphicsUnregisterResource(cudaResource));

#ifndef NDEBUG
    CRC::CoutDebug({{"Unregistered CUDA resource."}});
#endif
}

CRC_API void CRC::UnregisterCudaResource
(
    cudaGraphicsResource_t &cudaResource, Microsoft::WRL::ComPtr<ID3D11Device> &d3d11Device
){
    CRC::CheckCuda(cudaGraphicsUnregisterResource(cudaResource));
    HRESULT hr = d3d11Device->GetDeviceRemovedReason();
    if (FAILED(hr))
    {
#ifndef NDEBUG
        CRC::CoutErr({"Failed to unregister CUDA resource."});
#endif
        throw std::runtime_error("Failed to unregister CUDA resource.");
    }
}

CRC_API void CRC::UnregisterSwapChain3Presented
(
    std::vector<cudaGraphicsResource_t> &cudaResources, 
    Microsoft::WRL::ComPtr<ID3D11Device> &d3d11Device, Microsoft::WRL::ComPtr<IDXGISwapChain> &d3d11SwapChain, 
    UINT &frameIndex
){
    UINT bufferCount = cudaResources.size();
    if (bufferCount != 3)
    {
#ifndef NDEBUG
        CRC::CoutErr({"This function is only for 3 buffer swap chains."});
#endif
        throw std::runtime_error("This function is only for 3 buffer swap chains.");
    }

    HRESULT hr = S_OK;

    UINT targetIndex = (frameIndex + 1) % bufferCount;
    CRC::UnregisterCudaResource(cudaResources[targetIndex], d3d11Device);

    CRC::PresentD3D11SwapChain(d3d11SwapChain, 0, 0, bufferCount, frameIndex);

    targetIndex = (frameIndex + 2) % bufferCount;
    CRC::UnregisterCudaResource(cudaResources[targetIndex], d3d11Device);

    targetIndex = (frameIndex + 1) % bufferCount;
    CRC::UnregisterCudaResource(cudaResources[targetIndex], d3d11Device);

#ifndef NDEBUG
    CRC::CoutDebug({"Unregistered CUDA resources in swap chain."});
#endif
}

CRC_API void CRC::UnregisterSwapChain2Presented
(
    std::vector<cudaGraphicsResource_t> &cudaResources, 
    Microsoft::WRL::ComPtr<ID3D11Device> &d3d11Device, Microsoft::WRL::ComPtr<IDXGISwapChain> &d3d11SwapChain, 
    UINT &frameIndex
){
    UINT bufferCount = cudaResources.size();
    if (bufferCount != 2)
    {
#ifndef NDEBUG
        CRC::CoutErr({"This function is only for 2 buffer swap chains."});
#endif
        throw std::runtime_error("This function is only for 2 buffer swap chains.");
    }

    HRESULT hr = S_OK;

    UINT targetIndex = (frameIndex + 1) % bufferCount;
    CRC::UnregisterCudaResource(cudaResources[targetIndex], d3d11Device);

    CRC::PresentD3D11SwapChain(d3d11SwapChain, 0, 0, bufferCount, frameIndex);

    targetIndex = (frameIndex + 1) % bufferCount;
    CRC::UnregisterCudaResource(cudaResources[targetIndex], d3d11Device);

#ifndef NDEBUG
    CRC::CoutDebug({"Unregistered CUDA resources in swap chain."});
#endif
}

CRC_API void CRC::UnregisterSwapChainNotPresented
(
    std::vector<cudaGraphicsResource_t> &cudaResources, 
    Microsoft::WRL::ComPtr<ID3D11Device> &d3d11Device, Microsoft::WRL::ComPtr<IDXGISwapChain> &d3d11SwapChain, 
    UINT &frameIndex
){
    HRESULT hr = S_OK;
    for (int i = 0; i < cudaResources.size(); ++i) 
    {
        if (i == frameIndex) continue;
        CRC::UnregisterCudaResource(cudaResources[i], d3d11Device);
    }

    UINT targetIndex = frameIndex;

    CRC::PresentD3D11SwapChain(d3d11SwapChain, 0, 0, cudaResources.size(), frameIndex);

    CRC::UnregisterCudaResource(cudaResources[targetIndex], d3d11Device);

#ifndef NDEBUG
    CRC::CoutDebug({"Unregistered CUDA resources in swap chain."});
#endif
}

void CRC::MapCudaResource(cudaGraphicsResource_t& cudaResource, cudaStream_t stream)
{
    CRC::CheckCuda(cudaGraphicsMapResources(1, &cudaResource, stream));

#ifndef NDEBUG
    CRC::CoutDebug({"Mapped CUDA resource."});
#endif
}

void CRC::UnmapCudaResource(cudaGraphicsResource_t& cudaResource, cudaStream_t stream)
{
    CRC::CheckCuda(cudaGraphicsUnmapResources(1, &cudaResource, stream));

#ifndef NDEBUG
    CRC::CoutDebug({"Unmapped CUDA resource."});
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

    std::unique_ptr<CRCCudaResource> backTexture = std::make_unique<CRCCudaResource>(desc);
    backTexture->Assign(backBufferArray, CRC::GetBytesPerPixel(desc.Format) * desc.Width * desc.Height);

    CRC::UnmapCudaResource(cudaResource);
    return backTexture;
}

CRC_API ICRCTexture2D *CRC::CreatePtTexture2DFromCudaResource
(
    cudaGraphicsResource_t &cudaResource, D3D11_TEXTURE2D_DESC &desc
){
    CRC::MapCudaResource(cudaResource);
    cudaArray_t backBufferArray = CRC::GetCudaMappedArray(cudaResource);

    CRCCudaResource *backTexture = new CRCCudaResource(desc);
    backTexture->Assign(backBufferArray, CRC::GetBytesPerPixel(desc.Format) * desc.Width * desc.Height);

    CRC::UnmapCudaResource(cudaResource);
    return backTexture;
}

CRC_API void CRC::WaitForD3DGpuToFinish(Microsoft::WRL::ComPtr<ID3D11Device> &d3d11Device)
{
    ID3D11DeviceContext* d3d11DeviceContext = nullptr;
    d3d11Device->GetImmediateContext(&d3d11DeviceContext);

    d3d11DeviceContext->Flush();

    D3D11_QUERY_DESC queryDesc = {};
    queryDesc.Query = D3D11_QUERY_EVENT;
    ID3D11Query* pQuery = nullptr;
    d3d11Device->CreateQuery(&queryDesc, &pQuery);

    d3d11DeviceContext->End(pQuery);

    while (S_OK != d3d11DeviceContext->GetData(pQuery, nullptr, 0, 0)) 
    {
        Sleep(1);
    }

    pQuery->Release();
}

CRC_API void CRC::PresentD3D11SwapChain
(
    Microsoft::WRL::ComPtr<IDXGISwapChain> &d3d11SwapChain, UINT syncInterval, UINT flags, 
    const UINT &bufferCount, UINT &frameIndex
){
    HRESULT hr = d3d11SwapChain->Present(syncInterval, flags);
    if (FAILED(hr))
    {
#ifndef NDEBUG
        CRC::CoutErr({"Failed to present swap chain. IDXGISwapChain Present failed."});
#endif
        throw std::runtime_error("Failed to present swap chain. IDXGISwapChain Present failed.");
    }

    frameIndex = (frameIndex + 1) % bufferCount;
}
