#include "ExampleWndCtrl.h"

ExampleWndCtrl::ExampleWndCtrl()
{
#ifdef DEBUG_OUTPUT
    OutputDebugStringA("ExampleWndCtrl::ExampleWndCtrl()\n");

#endif // DEBUG_OUTPUT
}

ExampleWndCtrl::~ExampleWndCtrl()
{
#ifdef DEBUG_OUTPUT
    OutputDebugStringA("ExampleWndCtrl::~ExampleWndCtrl()\n");

#endif // DEBUG_OUTPUT
}

#ifdef DEBUG_OUTPUT

void ExampleWndCtrl::DebugInput()
{
    if (input->IsKeyDown(CRC_KEY_MSG_ESCAPE))
    {
        OutputDebugStringA("KeyDown ESC\n");
    }

    if (input->GetKeyText() != "")
    {
        OutputDebugStringA(input->GetKeyText().c_str());
        OutputDebugStringA("\n");
    }

    if (input->IsKeyDouble(CRC_KEY_MSG_A))
    {
        OutputDebugStringA("Double A\n");
    }

    if (input->IsMouse(CRC_MOUSE_MSG_LBTN))
    {
        OutputDebugStringA("LBTN\n");
    }

    if (input->IsMouseDouble(CRC_MOUSE_MSG_LBTN))
    {
        OutputDebugStringA("Double LBTN\n");
    }

    int mouseWheel = input->GetMouseWheelDelta();
    if (mouseWheel != 0)
    {
        OutputDebugStringA("Mouse Wheel: ");
        OutputDebugStringA(std::to_string(mouseWheel).c_str());
        OutputDebugStringA("\n");
    }
}

void ExampleWndCtrl::DebugScene()
{
    sceneCtrl->SetName("ExampleSceneEdit");
}

#endif // DEBUG_OUTPUT

HRESULT ExampleWndCtrl::OnCreate(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
#ifdef DEBUG_OUTPUT
    OutputDebugStringA("ExampleWndCtrl::OnCreate()\n");

#endif // DEBUG_OUTPUT
    return S_OK;
}

HRESULT ExampleWndCtrl::OnSetFocus(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
#ifdef DEBUG_OUTPUT
    OutputDebugStringA("ExampleWndCtrl::OnSetFocus()\n");

#endif // DEBUG_OUTPUT
    return S_OK;
}

HRESULT ExampleWndCtrl::OnKillFocus(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
#ifdef DEBUG_OUTPUT
    OutputDebugStringA("ExampleWndCtrl::OnKillFocus()\n");

#endif // DEBUG_OUTPUT
    return S_OK;
}

HRESULT ExampleWndCtrl::OnMinimize(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
#ifdef DEBUG_OUTPUT
    OutputDebugStringA("ExampleWndCtrl::OnMinimize()\n");

#endif // DEBUG_OUTPUT
    return S_OK;
}

HRESULT ExampleWndCtrl::OnMaximize(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
#ifdef DEBUG_OUTPUT
    OutputDebugStringA("ExampleWndCtrl::OnMaximize()\n");

#endif // DEBUG_OUTPUT
    return S_OK;
}

HRESULT ExampleWndCtrl::OnRestored(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
#ifdef DEBUG_OUTPUT
    OutputDebugStringA("ExampleWndCtrl::OnRestored()\n");

#endif // DEBUG_OUTPUT
    return S_OK;
}

HRESULT ExampleWndCtrl::OnPaint(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
#ifdef DEBUG_FRAME_OUTPUT
    OutputDebugStringA("ExampleWndCtrl::OnPaint()\n");

#endif // DEBUG_FRAME_OUTPUT

#ifdef DEBUG_OUTPUT
    DebugInput();
    DebugScene();

#endif // DEBUG_OUTPUT

    return S_OK;
}

HRESULT ExampleWndCtrl::OnMove(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
#ifdef DEBUG_OUTPUT
    OutputDebugStringA("ExampleWndCtrl::OnMove()\n");

#endif // DEBUG_OUTPUT
    return S_OK;
}

HRESULT ExampleWndCtrl::OnClose(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
#ifdef DEBUG_OUTPUT
    OutputDebugStringA("ExampleWndCtrl::OnClose()\n");

#endif // DEBUG_OUTPUT
    return S_OK;
}

HRESULT ExampleWndCtrl::OnDestroy(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
#ifdef DEBUG_OUTPUT
    OutputDebugStringA("ExampleWndCtrl::OnDestroy()\n");

#endif // DEBUG_OUTPUT
    PostQuitMessage(0);
    return S_OK;
}

HRESULT ExampleWndCtrl::OnKeyDown(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
#ifdef DEBUG_OUTPUT
    OutputDebugStringA("ExampleWndCtrl::OnKeyDown()\n");

#endif // DEBUG_OUTPUT
    return S_OK;
}

HRESULT ExampleWndCtrl::OnKeyUp(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
#ifdef DEBUG_OUTPUT
    OutputDebugStringA("ExampleWndCtrl::OnKeyUp()\n");

#endif // DEBUG_OUTPUT
    return S_OK;
}

HRESULT ExampleWndCtrl::OnMouse(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
#ifdef DEBUG_FRAME_OUTPUT
    OutputDebugStringA("ExampleWndCtrl::OnMouse()\n");

#endif // DEBUG_FRAME_OUTPUT

#ifdef DEBUG_OUTPUT

#endif // DEBUG_OUTPUT
    return S_OK;
}
