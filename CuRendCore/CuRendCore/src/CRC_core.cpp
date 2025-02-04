#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_core.h"

#include "CRC_window.h"
#include "CRC_scene.h"

CRCCore::CRCCore()
{
}

CRCCore::~CRCCore()
{
}

void CRCCore::Initialize()
{
    containers_.resize(CRC::CORE_CONTAINER_COUNT);
}

int CRCCore::Shutdown()
{
    CRC::Core() = nullptr;
    return 0;
}

HRESULT CRCCore::SetWindowContainer(std::unique_ptr<ICRCContainer> container)
{
    CRCWindowContainer* windowContainer = CRC::PtrAs<CRCWindowContainer>(container.get());

    if (windowContainer) containers_[CRC::ID_WINDOW_CONTAINER] = std::move(container);
    else return E_FAIL;
}

HRESULT CRCCore::SetSceneContainer(std::unique_ptr<ICRCContainer> container)
{
    CRCSceneContainer* sceneContainer = CRC::PtrAs<CRCSceneContainer>(container.get());

    if (sceneContainer) containers_[CRC::ID_SCENE_CONTAINER] = std::move(container);
    else return E_FAIL;
}

HRESULT CRCCore::CreateWindowCRC(int idWindow, std::unique_ptr<ICRCPhaseMethod> phaseMethod)
{
    if(
        containers_[CRC::ID_WINDOW_CONTAINER] == nullptr || 
        idWindow == CRC::ID_INVALID || 
        idWindow >= containers_[CRC::ID_WINDOW_CONTAINER]->GetSize()
    ) return E_FAIL;

    CRCWindowAttr* attr = CRC::PtrAs<CRCWindowAttr>(containers_[CRC::ID_WINDOW_CONTAINER]->Get(idWindow));

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

    // Awake window's phase method.
    phaseMethod->Awake();

    // Add window to existWindows_.
    existWindows_[attr->hWnd_] = std::make_pair(containers_[CRC::ID_WINDOW_CONTAINER].get(), std::move(phaseMethod));

    // Release source.
    attr->src_.reset();

    return S_OK;
}

HRESULT CRCCore::ShowWindowCRC(int idWindow)
{
    if(
        containers_[CRC::ID_WINDOW_CONTAINER] == nullptr || 
        idWindow == CRC::ID_INVALID || 
        idWindow >= containers_[CRC::ID_WINDOW_CONTAINER]->GetSize()
    ) return E_FAIL;

    CRCWindowAttr* windowData = CRC::PtrAs<CRCWindowAttr>(containers_[CRC::ID_WINDOW_CONTAINER]->Get(idWindow));

    if (!windowData->hWnd_) return E_FAIL;

    ShowWindow(windowData->hWnd_, SW_SHOW);
    UpdateWindow(windowData->hWnd_);

    return S_OK;
}

HRESULT CRCCore::CreateScene(int idScene, std::unique_ptr<ICRCPhaseMethod> phaseMethod)
{
    if(
        containers_[CRC::ID_SCENE_CONTAINER] == nullptr || 
        idScene == CRC::ID_INVALID || 
        idScene >= containers_[CRC::ID_SCENE_CONTAINER]->GetSize()
    ) return E_FAIL;

    //TODO: Implement scene creation.

    return S_OK;
}

HRESULT CRCCore::SetSceneToWindow(int idWindow, int idScene)
{
    if(
        containers_[CRC::ID_WINDOW_CONTAINER] == nullptr || containers_[CRC::ID_SCENE_CONTAINER] == nullptr ||
        idWindow == CRC::ID_INVALID || idScene == CRC::ID_INVALID ||
        idWindow >= containers_[CRC::ID_WINDOW_CONTAINER]->GetSize() || 
        idScene >= containers_[CRC::ID_SCENE_CONTAINER]->GetSize()
    ) return E_FAIL;

    CRCWindowAttr* windowData = CRC::PtrAs<CRCWindowAttr>(containers_[CRC::ID_WINDOW_CONTAINER]->Get(idWindow));
    windowData->idScene_ = idScene;

    return S_OK;
}

void CRCCore::HandleWindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (existWindows_.find(hWnd) == existWindows_.end()) return;

    switch (msg)
    {
    case WM_DESTROY:
        existWindows_[hWnd].second->End();
        PostQuitMessage(0);
        break;

    case WM_SHOWWINDOW:
        if (wParam) existWindows_[hWnd].second->Show();
        else existWindows_[hWnd].second->Hide();
        break;

    case WM_SIZE:
        if (wParam == SIZE_MINIMIZED) existWindows_[hWnd].second->Hide();
        else if (wParam == SIZE_RESTORED) existWindows_[hWnd].second->Show();
        break;

    case WM_PAINT:
        existWindows_[hWnd].second->Update();
        break;

    default:
        break;
    }
}
