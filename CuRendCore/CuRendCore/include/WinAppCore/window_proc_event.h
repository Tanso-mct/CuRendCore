#pragma once
#include "config.h"

#include "event.h"
#include "container.h"

namespace WACore
{

class WIN_APP_CORE IWPEvent
{
public:
    virtual ~IWPEvent() = default;

    virtual void OnSetFocus(std::unique_ptr<IContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual void OnKillFocus(std::unique_ptr<IContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual void OnSize(std::unique_ptr<IContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual void OnUpdate(std::unique_ptr<IContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual void OnMove(std::unique_ptr<IContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual void OnDestroy(std::unique_ptr<IContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual void OnKeyDown(std::unique_ptr<IContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual void OnKeyUp(std::unique_ptr<IContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual void OnMouse(std::unique_ptr<IContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam);
};

using WPEventCaller = EventCaller<HWND, UINT, IWPEvent, std::unique_ptr<IContainer>&, UINT, WPARAM, LPARAM>;

void WIN_APP_CORE AddWPEventInsts
(
    HWND hWnd, std::unique_ptr<IWPEvent> inst,
    std::unique_ptr<IEventInstTable<HWND, IWPEvent>>& instTable
);

void WIN_APP_CORE AddWPEventFuncs
(
    std::unique_ptr
    <
        IEventFuncTable
        <
            UINT, IWPEvent, 
            std::unique_ptr<IContainer>&, UINT, WPARAM, LPARAM
        >
    >& funcTable
);

} // namespace WACore
