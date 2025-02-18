#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_window.h"
#include "CRC_scene.h"

#include "CRC_container.h"
#include "CRC_event.h"

CRC_API std::unique_ptr<ICRCContainable> CRC::CreateWindowAttr(std::unique_ptr<CRCWindowSrc> attr)
{
    std::unique_ptr<CRCWindowAttr> windowData = std::make_unique<CRCWindowAttr>();

    windowData->src_ = std::move(attr);
    return windowData;
}

CRC_API std::unique_ptr<ICRCContainable> CRC::CreateSceneAttr(std::unique_ptr<CRCSceneSrc> attr)
{
    std::unique_ptr<CRCSceneAttr> sceneData = std::make_unique<CRCSceneAttr>();

    sceneData->src_ = std::move(attr);
    return sceneData;
}

HRESULT CRC::CreateWindowCRC(std::unique_ptr<ICRCContainable> &windowAttr)
{
    if (!windowAttr) return E_FAIL;

    CRCWindowAttr* attr = CRC::PtrAs<CRCWindowAttr>(windowAttr.get());
    if (!attr) return E_FAIL;

    if (!RegisterClassEx(&attr->src_->wcex_)) return E_FAIL;

    attr->hWnd_ = CreateWindow
    (
        attr->src_->wcex_.lpszClassName,
        attr->src_->name_,
        attr->src_->style_,
        attr->src_->initialPosX_,
        attr->src_->initialPosY_,
        attr->src_->width_,
        attr->src_->height_,
        attr->src_->hWndParent_,
        nullptr,
        attr->src_->hInstance,
        nullptr
    );
    if (!attr->hWnd_) return E_FAIL;

    // Release source.
    attr->src_.reset();
    return S_OK;
}

HRESULT CRC::ShowWindowCRC(std::unique_ptr<ICRCContainable> &windowAttr)
{
    CRCWindowAttr* attr = CRC::PtrAs<CRCWindowAttr>(windowAttr.get());

    if (!attr->hWnd_) return E_FAIL;

    ShowWindow(attr->hWnd_, SW_SHOW);
    UpdateWindow(attr->hWnd_);

    return S_OK;
}

HRESULT CRC::CreateScene(std::unique_ptr<ICRCContainable> &sceneAttr)
{
    if (!sceneAttr) return E_FAIL;

    CRCSceneAttr* attr = CRC::PtrAs<CRCSceneAttr>(sceneAttr.get());

    //TODO: Implement creating scene.

    // Release source.
    attr->src_.reset();
    return S_OK;
}

HRESULT CRC::CreateSwapChain(std::unique_ptr<ICRCContainable> &windowAttr)
{
    CRCWindowAttr* attr = CRC::PtrAs<CRCWindowAttr>(windowAttr.get());
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
