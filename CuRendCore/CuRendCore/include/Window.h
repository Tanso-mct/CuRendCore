#pragma once

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

#include "Math.cuh"
#include "Input.h"
#include "Scene.h"

namespace CRC 
{

class CRC_API WindowController;

typedef struct CRC_API _WINDOW_ATTRIBUTES
{
    WNDCLASSEX wcex = { 0 };
    LPCWSTR name = CRC_WND_DEFAULT_NAME;
    int initialPosX = CRC_WND_DEFAULT_POS_X;
    int initialPosY = CRC_WND_DEFAULT_POS_Y;
    unsigned int width = CRC_WND_DEFAULT_WIDTH;
    unsigned int height = CRC_WND_DEFAULT_HEIGHT;
    DWORD style = WS_OVERLAPPEDWINDOW;
    HWND hWndParent = NULL;
    HINSTANCE hInstance = nullptr;
    std::shared_ptr<WindowController> ctrl = nullptr;
} WNDATTR;


class CRC_API Window
{
private:
    Window(WNDATTR wattr);

    HWND hWnd = NULL;
    WNDATTR wattr = { 0 };
    std::shared_ptr<WindowController> ctrl = nullptr;

public:
    ~Window();

    friend class WindowFactory;
};

class CRC_API WindowController
{
protected:
    Input* input = nullptr;
    std::shared_ptr<SceneController> sceneCtrl = nullptr;

public:
    WindowController();
    virtual ~WindowController() = default;

    virtual HRESULT OnCreate(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam){ return S_OK; };
    virtual HRESULT OnSetFocus(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam){ return S_OK; };
    virtual HRESULT OnKillFocus(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam){ return S_OK; };
    virtual HRESULT OnMinimize(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam){ return S_OK; };
    virtual HRESULT OnMaximize(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam){ return S_OK; };
    virtual HRESULT OnRestored(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam){ return S_OK; };
    virtual HRESULT OnPaint(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam){ return S_OK; };
    virtual HRESULT OnMove(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam){ return S_OK; };
    virtual HRESULT OnClose(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam){ return S_OK; };
    virtual HRESULT OnDestroy(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam){PostQuitMessage(0); return S_OK; };
    virtual HRESULT OnKeyDown(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam){ return S_OK; };
    virtual HRESULT OnKeyUp(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam){ return S_OK; };
    virtual HRESULT OnMouse(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam){ return S_OK; };

    friend class WindowFactory;
};

class CRC_API WindowFactory
{
private:
    WindowFactory() = default;
    std::vector<std::shared_ptr<Window>> windows;
    std::unordered_map<HWND, CRC_SLOT> slots;

    std::shared_ptr<Window> creatingWnd;

public:
    ~WindowFactory();

    static WindowFactory* GetInstance();
    static void ReleaseInstance();

    CRC_SLOT CreateWindowCRC(WNDATTR wattr);
    HRESULT DestroyWindowCRC(CRC_SLOT slot);
    HRESULT ShowWindowCRC(CRC_SLOT slot, int nCmdShow);

    HRESULT SetSceneCtrl(CRC_SLOT slot, std::shared_ptr<SceneController> sceneCtrl);

    static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
};

} // namespace CRC