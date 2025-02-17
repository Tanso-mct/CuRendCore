#pragma once

#include "CRC_config.h"
#include "CRC_container.h"

#include <Windows.h>

class ICRCWinMsgEvent
{
public:
    virtual ~ICRCWinMsgEvent() = default;

    virtual void OnSetFocus(UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnKillFocus(UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnSize(UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnUpdate(UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnMove(UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnDestroy(UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnKeyDown(UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnKeyUp(UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnMouse(UINT msg, WPARAM wParam, LPARAM lParam){};
};