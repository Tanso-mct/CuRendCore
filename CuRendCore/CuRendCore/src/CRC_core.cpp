#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_core.h"

#include "CRC_window.h"
#include "CRC_scene.h"
#include "CRC_container.h"
#include "CRC_phase_method.h"

CRCCore::CRCCore()
{
}

CRCCore::~CRCCore()
{
}

void CRCCore::Initialize()
{
    containers_ = std::make_unique<CRCContainerSet>();
}

int CRCCore::Shutdown()
{
    CRC::Core() = nullptr;
    return 0;
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
