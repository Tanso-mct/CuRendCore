#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_core.h"

#include "CRC_window.h"
#include "CRC_scene.h"
#include "CRC_container.h"
#include "CRC_event.h"

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
    handledMsgMap_[WM_SETFOCUS] = &ICRCWinMsgEvent::OnSetFocus;
    handledMsgMap_[WM_KILLFOCUS] = &ICRCWinMsgEvent::OnKillFocus;
    handledMsgMap_[WM_SIZE] = &ICRCWinMsgEvent::OnSize;
    handledMsgMap_[WM_MOVE] = &ICRCWinMsgEvent::OnMove;
    handledMsgMap_[WM_DESTROY] = &ICRCWinMsgEvent::OnDestroy;

    handledMsgMap_[WM_KEYDOWN] = &ICRCWinMsgEvent::OnKeyDown;
    handledMsgMap_[WM_SYSKEYDOWN] = &ICRCWinMsgEvent::OnKeyDown;
    handledMsgMap_[WM_KEYUP] = &ICRCWinMsgEvent::OnKeyUp;
    handledMsgMap_[WM_SYSKEYUP] = &ICRCWinMsgEvent::OnKeyUp;

    handledMsgMap_[WM_LBUTTONDOWN] = &ICRCWinMsgEvent::OnMouse;
    handledMsgMap_[WM_LBUTTONUP] = &ICRCWinMsgEvent::OnMouse;
    handledMsgMap_[WM_RBUTTONDOWN] = &ICRCWinMsgEvent::OnMouse;
    handledMsgMap_[WM_RBUTTONUP] = &ICRCWinMsgEvent::OnMouse;
    handledMsgMap_[WM_MBUTTONDOWN] = &ICRCWinMsgEvent::OnMouse;
    handledMsgMap_[WM_MBUTTONUP] = &ICRCWinMsgEvent::OnMouse;
    handledMsgMap_[WM_MOUSEWHEEL] = &ICRCWinMsgEvent::OnMouse;
    handledMsgMap_[WM_MOUSEMOVE] = &ICRCWinMsgEvent::OnMouse;

    // Initialize window message caller.
    winMsgCaller_ = std::make_unique<CRCEventCaller<HWND, ICRCWinMsgEvent, UINT, WPARAM, LPARAM>>();
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
    winMsgCaller_->Call(hWnd, &ICRCWinMsgEvent::OnUpdate, msg, 0, 0);
}
