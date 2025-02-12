#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_core.h"

#include "CRC_window.h"
#include "CRC_scene.h"
#include "CRC_container.h"

CRCCore::CRCCore()
{
}

CRCCore::~CRCCore()
{
}

void CRCCore::Initialize()
{
    containerSet_ = std::make_unique<CRCContainerSet>();
    pmCaller_ = std::make_unique<CRCPhaseMethodCaller<HWND>>();
}

int CRCCore::Shutdown()
{
    CRC::Core() = nullptr;
    return 0;
}

void CRCCore::HandleWindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_DESTROY:
        pmCaller_->CallEnd(hWnd);
        PostQuitMessage(0);
        return;

    case WM_SIZE:
        if (wParam == SIZE_MINIMIZED) pmCaller_->CallHide(hWnd);
        else if (wParam == SIZE_RESTORED) pmCaller_->CallRestored(hWnd);
        return;

    case WM_PAINT:
        pmCaller_->CallUpdate(hWnd);
        return;

    default:
        return;
    }
}
