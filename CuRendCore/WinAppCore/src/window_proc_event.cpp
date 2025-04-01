#include "pch.h"
#include "window_proc_event.h"

void WIN_APP_CORE WACore::AddWPEventInsts
(
    HWND hWnd, std::unique_ptr<IWPEvent> inst, 
    std::unique_ptr<IEventInstTable<HWND, IWPEvent>> &instTable
){
    instTable->Add(hWnd, std::move(inst));
}

void WIN_APP_CORE WACore::AddWPEventFuncs
(
    std::unique_ptr<WACore::IEventFuncTable
    <
        UINT, WACore::IWPEvent,
        std::unique_ptr<WACore::IContainer> &, UINT, WPARAM, LPARAM
    >> &funcTable
){
    funcTable->Add(WM_SETFOCUS, &IWPEvent::OnSetFocus);
    funcTable->Add(WM_KILLFOCUS, &IWPEvent::OnKillFocus);
    funcTable->Add(WM_PAINT, &IWPEvent::OnUpdate);
    funcTable->Add(WM_SIZE, &IWPEvent::OnSize);
    funcTable->Add(WM_PAINT, &IWPEvent::OnUpdate);
    funcTable->Add(WM_MOVE, &IWPEvent::OnMove);
    funcTable->Add(WM_DESTROY, &IWPEvent::OnDestroy);

    funcTable->Add(WM_KEYDOWN, &IWPEvent::OnKeyDown);
    funcTable->Add(WM_SYSKEYDOWN, &IWPEvent::OnKeyDown);
    funcTable->Add(WM_KEYUP, &IWPEvent::OnKeyUp);
    funcTable->Add(WM_SYSKEYUP, &IWPEvent::OnKeyUp);

    funcTable->Add(WM_LBUTTONDOWN, &IWPEvent::OnMouse);
    funcTable->Add(WM_LBUTTONUP, &IWPEvent::OnMouse);
    funcTable->Add(WM_RBUTTONDOWN, &IWPEvent::OnMouse);
    funcTable->Add(WM_RBUTTONUP, &IWPEvent::OnMouse);
    funcTable->Add(WM_MBUTTONDOWN, &IWPEvent::OnMouse);
    funcTable->Add(WM_MBUTTONUP, &IWPEvent::OnMouse);
    funcTable->Add(WM_MOUSEWHEEL, &IWPEvent::OnMouse);
    funcTable->Add(WM_MOUSEMOVE, &IWPEvent::OnMouse);
}

void WACore::IWPEvent::OnSetFocus(std::unique_ptr<IContainer> &container, UINT msg, WPARAM wParam, LPARAM lParam)
{
}

void WACore::IWPEvent::OnKillFocus(std::unique_ptr<IContainer> &container, UINT msg, WPARAM wParam, LPARAM lParam)
{
}

void WACore::IWPEvent::OnSize(std::unique_ptr<IContainer> &container, UINT msg, WPARAM wParam, LPARAM lParam)
{
}

void WACore::IWPEvent::OnUpdate(std::unique_ptr<IContainer> &container, UINT msg, WPARAM wParam, LPARAM lParam)
{
}

void WACore::IWPEvent::OnMove(std::unique_ptr<IContainer> &container, UINT msg, WPARAM wParam, LPARAM lParam)
{
}

void WACore::IWPEvent::OnDestroy(std::unique_ptr<IContainer> &container, UINT msg, WPARAM wParam, LPARAM lParam)
{
}

void WACore::IWPEvent::OnKeyDown(std::unique_ptr<IContainer> &container, UINT msg, WPARAM wParam, LPARAM lParam)
{
}

void WACore::IWPEvent::OnKeyUp(std::unique_ptr<IContainer> &container, UINT msg, WPARAM wParam, LPARAM lParam)
{
}

void WACore::IWPEvent::OnMouse(std::unique_ptr<IContainer> &container, UINT msg, WPARAM wParam, LPARAM lParam)
{
}
