#pragma once

// #define DEBUG_OUTPUT
// #define DEBUG_FRAME_OUTPUT

#include "CuRendCore.h"

class ExampleWndCtrl : public CRC::WindowController
{
public:
    ExampleWndCtrl();
    ~ExampleWndCtrl() override;

#ifdef DEBUG_OUTPUT
    void DebugInput();
    void DebugScene();

#endif // DEBUG_OUTPUT

    HRESULT OnCreate(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) override;
    HRESULT OnSetFocus(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) override;
    HRESULT OnKillFocus(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) override;
    HRESULT OnMinimize(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) override;
    HRESULT OnMaximize(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) override;
    HRESULT OnRestored(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) override;
    HRESULT OnPaint(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) override;
    HRESULT OnMove(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) override;
    HRESULT OnClose(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) override;
    HRESULT OnDestroy(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) override;
    HRESULT OnKeyDown(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) override;
    HRESULT OnKeyUp(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) override;
    HRESULT OnMouse(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) override;
};

