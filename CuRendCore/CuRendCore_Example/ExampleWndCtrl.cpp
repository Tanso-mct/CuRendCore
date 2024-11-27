#include "ExampleWndCtrl.h"

ExampleWndCtrl::ExampleWndCtrl()
{
    OutputDebugStringA("ExampleWndCtrl::ExampleWndCtrl()\n");
}

ExampleWndCtrl::~ExampleWndCtrl()
{
    OutputDebugStringA("ExampleWndCtrl::~ExampleWndCtrl()\n");
}

HRESULT ExampleWndCtrl::OnCreate()
{
    OutputDebugStringA("ExampleWndCtrl::OnCreate()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnSetFocus()
{
    OutputDebugStringA("ExampleWndCtrl::OnSetFocus()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnKillFocus()
{
    OutputDebugStringA("ExampleWndCtrl::OnKillFocus()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnMinimize()
{
    OutputDebugStringA("ExampleWndCtrl::OnMinimize()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnMaximize()
{
    OutputDebugStringA("ExampleWndCtrl::OnMaximize()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnRestored()
{
    OutputDebugStringA("ExampleWndCtrl::OnRestored()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnPaint()
{
    // OutputDebugStringA("ExampleWndCtrl::OnPaint()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnMove()
{
    OutputDebugStringA("ExampleWndCtrl::OnMove()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnClose()
{
    OutputDebugStringA("ExampleWndCtrl::OnClose()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnDestroy()
{
    OutputDebugStringA("ExampleWndCtrl::OnDestroy()\n");
    PostQuitMessage(0);
    return S_OK;
}

HRESULT ExampleWndCtrl::OnKeyDown()
{
    OutputDebugStringA("ExampleWndCtrl::OnKeyDown()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnKeyUp()
{
    OutputDebugStringA("ExampleWndCtrl::OnKeyUp()\n");
    return S_OK;
}

HRESULT ExampleWndCtrl::OnMouse()
{
    // OutputDebugStringA("ExampleWndCtrl::OnMouse()\n");
    return S_OK;
}
