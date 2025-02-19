#include "CRC_pch.h"
#include "CRC_funcs.h"

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

HRESULT CRC::CreateSwapChain(std::unique_ptr<ICRCContainable> &windowAttr)
{
    CRCWindowAttr* attr = CRC::As<CRCWindowAttr>(windowAttr.get());
    if (!attr) return E_FAIL;
    if (!attr->hWnd_) return E_FAIL;

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
    sd.OutputWindow = attr->hWnd_;
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
        D3D11_SDK_VERSION, &sd, &attr->swapChain_, &attr->device_, &featureLevel, nullptr
    );

    if (hr == DXGI_ERROR_UNSUPPORTED) // Try high-performance WARP software driver if hardware is not available.
    {
        hr = D3D11CreateDeviceAndSwapChain
        (
            nullptr, D3D_DRIVER_TYPE_WARP, nullptr, createDeviceFlags, featureLevelArray, 2, 
            D3D11_SDK_VERSION, &sd, &attr->swapChain_, &attr->device_, &featureLevel, nullptr
        );
    }

    return hr;
}
