#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_core.h"

#include "CRC_window.h"
#include "CRC_scene.h"
#include "CRC_container.h"
#include "CRC_event_listener.h"

CRCCore::CRCCore()
{
}

CRCCore::~CRCCore()
{
}

void CRCCore::Initialize()
{
    containerSet_ = std::make_unique<CRCContainerSet>();
    winMsgCaller_ = std::make_unique<CRCEventCaller<HWND, ICRCWinMsgListener, UINT, WPARAM, LPARAM>>();
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
        winMsgCaller_->Call(hWnd, &ICRCWinMsgListener::OnDestroy, msg, wParam, lParam);
        return;

    case WM_SIZE:
        if (wParam == SIZE_MINIMIZED)
        {
            winMsgCaller_->Call(hWnd, &ICRCWinMsgListener::OnMinimize, msg, wParam, lParam);
        }
        else if (wParam == SIZE_MAXIMIZED)
        {
            winMsgCaller_->Call(hWnd, &ICRCWinMsgListener::OnMaximize, msg, wParam, lParam);
        }
        else if (wParam == SIZE_RESTORED)
        {
            winMsgCaller_->Call(hWnd, &ICRCWinMsgListener::OnRestored, msg, wParam, lParam);
        }

        return;

    default:
        return;
    }
}

void CRCCore::FrameUpdate(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    winMsgCaller_->Call(hWnd, &ICRCWinMsgListener::OnUpdate, WM_PAINT, 0, 0);
}
