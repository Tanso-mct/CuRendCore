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

HRESULT CRCCore::CreateWindowCRC(int idWindow, ICRCPhaseMethod* phaseMethod)
{
    if(
        containers_[CRC::ID_WINDOW_CONTAINER] == nullptr || 
        idWindow == CRC::ID_INVALID || 
        idWindow >= containers_[CRC::ID_WINDOW_CONTAINER]->GetSize() ||
        phaseMethod == nullptr
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
    existWindows_[attr->hWnd_] = std::make_tuple(attr, phaseMethod, nullptr);

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

    CRCWindowAttr* windowAttr = CRC::PtrAs<CRCWindowAttr>(containers_[CRC::ID_WINDOW_CONTAINER]->Get(idWindow));

    if (!windowAttr->hWnd_) return E_FAIL;

    ShowWindow(windowAttr->hWnd_, SW_SHOW);
    UpdateWindow(windowAttr->hWnd_);

    return S_OK;
}

HRESULT CRCCore::CreateScene(int idScene, ICRCPhaseMethod* phaseMethod)
{
    if(
        containers_[CRC::ID_SCENE_CONTAINER] == nullptr || 
        idScene == CRC::ID_INVALID || 
        idScene >= containers_[CRC::ID_SCENE_CONTAINER]->GetSize() ||
        phaseMethod == nullptr
    ) return E_FAIL;

    CRCSceneAttr* sceneAttr = CRC::PtrAs<CRCSceneAttr>(containers_[CRC::ID_SCENE_CONTAINER]->Get(idScene));

    if (sceneAttr->isAwaked_) return S_OK; // Already awaked.

    sceneAttr->phaseMethod_ = phaseMethod;
    sceneAttr->phaseMethod_->Awake();
    sceneAttr->isAwaked_ = true;

    // Release source.
    sceneAttr->src_.reset();
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

    // Set scene id to window.
    CRCWindowAttr* windowAttr = CRC::PtrAs<CRCWindowAttr>(containers_[CRC::ID_WINDOW_CONTAINER]->Get(idWindow));
    windowAttr->idScene_ = idScene;

    // hWnd_ must be exist.
    if (existWindows_.find(windowAttr->hWnd_) == existWindows_.end()) return E_FAIL;

    // Set scene phase method.
    CRCSceneAttr* sceneAttr = CRC::PtrAs<CRCSceneAttr>(containers_[CRC::ID_SCENE_CONTAINER]->Get(idScene));
    std::get<2>(existWindows_[windowAttr->hWnd_]) = sceneAttr->phaseMethod_;

    return S_OK;
}

void CRCCore::HandleWindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (existWindows_.find(hWnd) == existWindows_.end()) return;

    switch (msg)
    {
    case WM_DESTROY:
        if (std::get<2>(existWindows_[hWnd])) std::get<2>(existWindows_[hWnd])->End();
        std::get<1>(existWindows_[hWnd])->End();
        PostQuitMessage(0);
        break;

    case WM_SIZE:
        if (wParam == SIZE_MINIMIZED)
        {
            if (std::get<2>(existWindows_[hWnd])) std::get<2>(existWindows_[hWnd])->Hide();
            std::get<1>(existWindows_[hWnd])->Hide();
        }
        else if (wParam == SIZE_RESTORED)
        {
            if (std::get<2>(existWindows_[hWnd])) std::get<2>(existWindows_[hWnd])->Show();
            std::get<1>(existWindows_[hWnd])->Show();
        }
        break;

    case WM_PAINT:
        if (std::get<2>(existWindows_[hWnd])) std::get<2>(existWindows_[hWnd])->Update();
        std::get<1>(existWindows_[hWnd])->Update();
        break;

    default:
        break;
    }
}
