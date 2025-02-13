#pragma once

#include "CRC_config.h"
#include "CRC_container.h"

#include <Windows.h>

class ICRCWinMsgListener
{
public:
    virtual ~ICRCWinMsgListener() = default;

    virtual void OnSetFocus(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnKillFocus(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnSize(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnUpdate(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnMove(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnDestroy(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnKeyDown(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnKeyUp(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnMouse(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam){};
};