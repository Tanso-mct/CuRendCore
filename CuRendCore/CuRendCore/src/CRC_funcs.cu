﻿#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_window.h"
#include "CRC_scene.h"

#include "CRC_container.h"
#include "CRC_event.h"

HRESULT CRC::ShowWindowCRC(HWND& hWnd)
{
    if (!hWnd) return E_FAIL;

    ShowWindow(hWnd, SW_SHOW);
    UpdateWindow(hWnd);

    return S_OK;
}

CRC_API HRESULT CRC::CreateDeviceAndSwapChain
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
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

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
