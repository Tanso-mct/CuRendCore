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
    // Initialize container set.
    containerSet_ = std::make_unique<CRCContainerSet>();

    // Initialize window message functions. Register a function that corresponds to the Window message.
    handledMsgMap_[WM_SETFOCUS] = &ICRCWinMsgListener::OnSetFocus;
    handledMsgMap_[WM_KILLFOCUS] = &ICRCWinMsgListener::OnKillFocus;
    handledMsgMap_[WM_SIZE] = &ICRCWinMsgListener::OnSize;
    handledMsgMap_[WM_MOVE] = &ICRCWinMsgListener::OnMove;
    handledMsgMap_[WM_DESTROY] = &ICRCWinMsgListener::OnDestroy;

    handledMsgMap_[WM_KEYDOWN] = &ICRCWinMsgListener::OnKeyDown;
    handledMsgMap_[WM_SYSKEYDOWN] = &ICRCWinMsgListener::OnKeyDown;
    handledMsgMap_[WM_KEYUP] = &ICRCWinMsgListener::OnKeyUp;
    handledMsgMap_[WM_SYSKEYUP] = &ICRCWinMsgListener::OnKeyUp;

    handledMsgMap_[WM_LBUTTONDOWN] = &ICRCWinMsgListener::OnMouse;
    handledMsgMap_[WM_LBUTTONUP] = &ICRCWinMsgListener::OnMouse;
    handledMsgMap_[WM_LBUTTONDBLCLK] = &ICRCWinMsgListener::OnMouse;

    handledMsgMap_[WM_RBUTTONDOWN] = &ICRCWinMsgListener::OnMouse;
    handledMsgMap_[WM_RBUTTONUP] = &ICRCWinMsgListener::OnMouse;
    handledMsgMap_[WM_RBUTTONDBLCLK] = &ICRCWinMsgListener::OnMouse;

    handledMsgMap_[WM_MBUTTONDOWN] = &ICRCWinMsgListener::OnMouse;
    handledMsgMap_[WM_MBUTTONUP] = &ICRCWinMsgListener::OnMouse;
    handledMsgMap_[WM_MBUTTONDBLCLK] = &ICRCWinMsgListener::OnMouse;
    handledMsgMap_[WM_MOUSEWHEEL] = &ICRCWinMsgListener::OnMouse;

    handledMsgMap_[WM_MOUSEMOVE] = &ICRCWinMsgListener::OnMouse;

    // Initialize window message caller.
    winMsgCaller_ = std::make_unique<CRCEventCaller<HWND, ICRCWinMsgListener, UINT, WPARAM, LPARAM>>();
}

int CRCCore::Shutdown()
{
    CRC::Core() = nullptr;
    return 0;
}

void CRCCore::HandleWindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (handledMsgMap_.find(msg) == handledMsgMap_.end()) return;
    winMsgCaller_->Call(hWnd, handledMsgMap_[msg], msg, wParam, lParam);
}

void CRCCore::FrameUpdate(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    winMsgCaller_->Call(hWnd, &ICRCWinMsgListener::OnUpdate, msg, 0, 0);
}
