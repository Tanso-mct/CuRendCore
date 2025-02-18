#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_event.h"

void CRC_API CRC::CreateWinMsgEventFuncMap(std::unordered_map<WinMsgEventKey, WinMsgEventFunc> &map)
{
    map[WM_SETFOCUS] = &ICRCWinMsgEvent::OnSetFocus;
    map[WM_KILLFOCUS] = &ICRCWinMsgEvent::OnKillFocus;
    map[WM_PAINT] = &ICRCWinMsgEvent::OnUpdate;
    map[WM_SIZE] = &ICRCWinMsgEvent::OnSize;
    map[WM_MOVE] = &ICRCWinMsgEvent::OnMove;
    map[WM_DESTROY] = &ICRCWinMsgEvent::OnDestroy;

    map[WM_KEYDOWN] = &ICRCWinMsgEvent::OnKeyDown;
    map[WM_SYSKEYDOWN] = &ICRCWinMsgEvent::OnKeyDown;
    map[WM_KEYUP] = &ICRCWinMsgEvent::OnKeyUp;
    map[WM_SYSKEYUP] = &ICRCWinMsgEvent::OnKeyUp;

    map[WM_LBUTTONDOWN] = &ICRCWinMsgEvent::OnMouse;
    map[WM_LBUTTONUP] = &ICRCWinMsgEvent::OnMouse;
    map[WM_RBUTTONDOWN] = &ICRCWinMsgEvent::OnMouse;
    map[WM_RBUTTONUP] = &ICRCWinMsgEvent::OnMouse;
    map[WM_MBUTTONDOWN] = &ICRCWinMsgEvent::OnMouse;
    map[WM_MBUTTONUP] = &ICRCWinMsgEvent::OnMouse;
    map[WM_MOUSEWHEEL] = &ICRCWinMsgEvent::OnMouse;
    map[WM_MOUSEMOVE] = &ICRCWinMsgEvent::OnMouse;
}
