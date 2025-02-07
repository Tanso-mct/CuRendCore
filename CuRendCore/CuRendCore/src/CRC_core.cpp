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

std::unique_ptr<ICRCContainer> &CRCCore::GetContainer(int id)
{
    if (id < 0 || id >= containers_.size()) return emptyContainer_;
    return containers_[id];
}

std::unique_ptr<ICRCContainer> CRCCore::TakeContainer(int id)
{
    if (id < 0 || id >= containers_.size()) return nullptr;
    return std::move(containers_[id]);
}

HRESULT CRCCore::PutContainer(int id, std::unique_ptr<ICRCContainer> container)
{
    if (id < 0 || id >= containers_.size()) return E_FAIL;
    containers_[id] = std::move(container);
}

HRESULT CRCCore::SetSceneToWindow(std::unique_ptr<ICRCContainable>& windowAttr, int idScene, int idSceneContainer)
{
    if(idSceneContainer < 0 || idSceneContainer >= containers_.size())
    {
        if (idScene < 0 || idScene >= containers_[idSceneContainer]->GetSize()) return E_FAIL;
    }

    // Set scene id to window.
    CRCWindowAttr* attr = CRC::PtrAs<CRCWindowAttr>(windowAttr.get());
    if (!attr) return E_FAIL;

    attr->idSceneContainer_ = idSceneContainer;
    attr->idScene_ = idScene;

    return S_OK;
}

HRESULT CRCCore::AddPhaseMethodToWindow(HWND hWnd, std::unique_ptr<ICRCPhaseMethod> phaseMethod)
{
    if (!phaseMethod) return E_FAIL;
    
    hWndPMs[hWnd].emplace_back(std::move(phaseMethod));
    return S_OK;
}

void CRCCore::HandleWindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (hWndPMs.find(hWnd) == hWndPMs.end()) return;

    switch (msg)
    {
    case WM_DESTROY:
        for (auto& pm : hWndPMs[hWnd]) pm->End();
        PostQuitMessage(0);
        return;

    case WM_SIZE:
        if (wParam == SIZE_MINIMIZED) for (auto& pm : hWndPMs[hWnd]) pm->Hide();
        else if (wParam == SIZE_RESTORED) for (auto& pm : hWndPMs[hWnd]) pm->Restored();
        return;

    case WM_PAINT:
        for (auto& pm : hWndPMs[hWnd]) pm->Update();
        return;

    default:
        return;
    }
}
