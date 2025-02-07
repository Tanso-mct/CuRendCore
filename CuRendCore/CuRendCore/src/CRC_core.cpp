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
}

int CRCCore::Shutdown()
{
    CRC::Core() = nullptr;
    return 0;
}

int CRCCore::AddContainer(std::unique_ptr<ICRCContainer> container)
{
    containers_.emplace_back(std::move(container));
    return containers_.size() - 1;
}

HRESULT CRCCore::CreateWindowCRC(int idWindow, int idWindowContainer, int idWindowPM, int idWindowPMContainer)
{
    if(
        idWindow < 0 || idWindow >= containers_[idWindowContainer]->GetSize() ||
        idWindowContainer < 0 || idWindowContainer >= containers_.size() ||

        idWindowPM < 0 || idWindowPM >= containers_[idWindowPMContainer]->GetSize() ||
        idWindowPMContainer < 0 || idWindowPMContainer >= containers_.size()
    ) return E_FAIL;

    CRCWindowAttr* attr = CRC::PtrAs<CRCWindowAttr>(containers_[idWindowContainer]->Get(idWindow));

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
    ICRCPhaseMethod* phaseMethod = CRC::PtrAs<ICRCPhaseMethod>(containers_[idWindowPMContainer]->Get(idWindowPM));
    phaseMethod->Awake();

    // Set window's phase method.
    hWndPMs[attr->hWnd_] = std::make_tuple(idWindowPMContainer, idWindowPM, CRC::ID_INVALID, CRC::ID_INVALID);

    // Release source.
    attr->src_.reset();
    return S_OK;
}

HRESULT CRCCore::ShowWindowCRC(int idWindow, int idWindowContainer)
{
    if(
        idWindow < 0 || idWindow >= containers_[idWindowContainer]->GetSize() ||
        idWindowContainer < 0 || idWindowContainer >= containers_.size()
    ) return E_FAIL;

    CRCWindowAttr* windowAttr = CRC::PtrAs<CRCWindowAttr>(containers_[idWindowContainer]->Get(idWindow));

    if (!windowAttr->hWnd_) return E_FAIL;

    ShowWindow(windowAttr->hWnd_, SW_SHOW);
    UpdateWindow(windowAttr->hWnd_);

    return S_OK;
}

HRESULT CRCCore::CreateScene(int idScene, int idSceneContainer, int idScenePM, int idScenePMContainer)
{
    if(
        idScene < 0 || idScene >= containers_[idSceneContainer]->GetSize() ||
        idSceneContainer < 0 || idSceneContainer >= containers_.size() ||
        
        idScenePM < 0 || idScenePM >= containers_[idScenePMContainer]->GetSize() ||
        idScenePMContainer < 0 || idScenePMContainer >= containers_.size()
    ) return E_FAIL;

    CRCSceneAttr* sceneAttr = CRC::PtrAs<CRCSceneAttr>(containers_[idSceneContainer]->Get(idScene));

    if (sceneAttr->isAwaked_) return S_OK; // Already awaked.

    // Awake scene's phase method.
    sceneAttr->phaseMethod_ = CRC::PtrAs<ICRCPhaseMethod>(containers_[idScenePMContainer]->Get(idScenePM));
    sceneAttr->phaseMethod_->Awake();
    sceneAttr->isAwaked_ = true;

    // Release source.
    sceneAttr->src_.reset();
    return S_OK;
}

HRESULT CRCCore::SetSceneToWindow(int idWindow, int idWindowContainer, int idScene, int idSceneContainer)
{
    if(
        idWindow < 0 || idWindow >= containers_[idWindowContainer]->GetSize() || 
        idWindowContainer < 0 || idWindowContainer >= containers_.size() ||

        idScene < 0 || idScene >= containers_[idSceneContainer]->GetSize() ||
        idSceneContainer < 0 || idSceneContainer >= containers_.size()
    ) return E_FAIL;

    // Set scene id to window.
    CRCWindowAttr* windowAttr = CRC::PtrAs<CRCWindowAttr>(containers_[idWindowContainer]->Get(idWindow));
    windowAttr->idScene_ = idScene;

    // Set scene phase method.
    CRCSceneAttr* sceneAttr = CRC::PtrAs<CRCSceneAttr>(containers_[idSceneContainer]->Get(idScene));
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
