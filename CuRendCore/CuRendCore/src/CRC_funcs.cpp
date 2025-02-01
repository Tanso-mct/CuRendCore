#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_core.h"
#include "CRC_window.h"
#include "CRC_scene.h"

CRC_API std::unique_ptr<CRCCore>& CRC::Core()
{
    static std::unique_ptr<CRCCore> core = std::make_unique<CRCCore>();
    return core;
}

CRC_API std::unique_ptr<CRCContainer> CRC::CreateWindowContainer()
{
    std::unique_ptr<CRCContainer> windowContainer = std::make_unique<CRCWindowContainer>();
    return windowContainer;
}

CRC_API std::unique_ptr<CRCData> CRC::CreateCRCWindow(CRCWindowAttr& attr)
{
    std::unique_ptr<CRCWindowData> windowData = std::make_unique<CRCWindowData>();

    if (!RegisterClassEx(&attr.wcex_)) return nullptr;

    windowData->hWnd_ = CreateWindow
    (
        attr.wcex_.lpszClassName,
        attr.name_,
        attr.style_,
        attr.initialPosX_,
        attr.initialPosY_,
        attr.width_,
        attr.height_,
        attr.hWndParent_,
        nullptr,
        attr.hInstance,
        nullptr
    );
    if (!windowData->hWnd_) return nullptr;

    return windowData;
}

CRC_API HRESULT CRC::ShowCRCWindow(std::unique_ptr<CRCData> &data)
{
    std::unique_ptr<CRCWindowData>& windowData = CastRef<CRCWindowData>(data);

    if (!windowData->hWnd_) return E_FAIL;

    HRESULT hr = ShowWindow(windowData->hWnd_, SW_SHOW);
    if (FAILED(hr)) return hr;

    hr = UpdateWindow(windowData->hWnd_);

    return hr;
}

CRC_API std::unique_ptr<CRCContainer> CRC::CreateSceneContainer()
{
    std::unique_ptr<CRCContainer> sceneContainer = std::make_unique<CRCSceneContainer>();
    return sceneContainer;
}

CRC_API std::unique_ptr<CRCData> CRC::CreateCRCScene(CRCSceneAttr &attr)
{
    std::unique_ptr<CRCSceneData> sceneData = std::make_unique<CRCSceneData>();

    // Set scene attributes.

    return sceneData;
}