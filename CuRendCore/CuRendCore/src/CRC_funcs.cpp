#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_core.h"
#include "CRC_window.h"
#include "CRC_scene.h"

#include "CRC_container.h"
#include "CRC_event_listener.h"

CRC_API std::unique_ptr<CRCCore>& CRC::Core()
{
    static std::unique_ptr<CRCCore> core = std::make_unique<CRCCore>();
    return core;
}

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