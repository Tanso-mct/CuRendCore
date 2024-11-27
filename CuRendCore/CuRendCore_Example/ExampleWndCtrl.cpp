#include "ExampleWndCtrl.h"

ExampleWndCtrl::ExampleWndCtrl()
{
    OutputDebugStringA("ExampleWndCtrl::ExampleWndCtrl()\n");
}

ExampleWndCtrl::~ExampleWndCtrl()
{
    OutputDebugStringA("ExampleWndCtrl::~ExampleWndCtrl()\n");
}

HRESULT ExampleWndCtrl::OnCreate(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    OutputDebugStringA("ExampleWndCtrl::OnCreate()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnSetFocus(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    OutputDebugStringA("ExampleWndCtrl::OnSetFocus()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnKillFocus(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    OutputDebugStringA("ExampleWndCtrl::OnKillFocus()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnMinimize(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    OutputDebugStringA("ExampleWndCtrl::OnMinimize()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnMaximize(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    OutputDebugStringA("ExampleWndCtrl::OnMaximize()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnRestored(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    OutputDebugStringA("ExampleWndCtrl::OnRestored()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnPaint(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    // OutputDebugStringA("ExampleWndCtrl::OnPaint()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnMove(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    OutputDebugStringA("ExampleWndCtrl::OnMove()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnClose(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    OutputDebugStringA("ExampleWndCtrl::OnClose()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnDestroy(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    OutputDebugStringA("ExampleWndCtrl::OnDestroy()\n");
    PostQuitMessage(0);
    return S_OK;
}

HRESULT ExampleWndCtrl::OnKeyDown(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    OutputDebugStringA("ExampleWndCtrl::OnKeyDown()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnKeyUp(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    OutputDebugStringA("ExampleWndCtrl::OnKeyUp()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnMouse(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    // OutputDebugStringA("ExampleWndCtrl::OnMouse()\n");
    return S_OK;
}
