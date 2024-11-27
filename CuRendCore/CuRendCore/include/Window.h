#pragma once

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <unordered_map>

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
public:
    WindowController() = default;
    virtual ~WindowController() = default;

    virtual HRESULT OnCreate(){ return S_OK; };
    virtual HRESULT OnSetFocus(){ return S_OK; };
    virtual HRESULT OnKillFocus(){ return S_OK; };
    virtual HRESULT OnMinimize(){ return S_OK; };
    virtual HRESULT OnMaximize(){ return S_OK; };
    virtual HRESULT OnRestored(){ return S_OK; };
    virtual HRESULT OnPaint(){ return S_OK; };
    virtual HRESULT OnMove(){ return S_OK; };
    virtual HRESULT OnClose(){ return S_OK; };
    virtual HRESULT OnDestroy(){PostQuitMessage(0); return S_OK; };
    virtual HRESULT OnKeyDown(){ return S_OK; };
    virtual HRESULT OnKeyUp(){ return S_OK; };
    virtual HRESULT OnMouse(){ return S_OK; };
};

class CRC_API WindowFactory
{
private:
    WindowFactory();
    std::vector<std::shared_ptr<Window>> windows;
    std::unordered_map<HWND, CRC_SLOT> slots;

    std::shared_ptr<Window> creatingWnd;

public:
    ~WindowFactory();
    static WindowFactory* GetInstance();

    CRC_SLOT CreateWindowCRC(WNDATTR wattr);
    HRESULT DestroyWindowCRC(CRC_SLOT slot);
    HRESULT ShowWindowCRC(CRC_SLOT slot, int nCmdShow);

    static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
};

} // namespace CRC