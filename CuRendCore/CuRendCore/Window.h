#pragma once

#include "CRCConfig.h"

#include <vector>
#include <memory>

namespace CRC 
{

typedef struct CRC_API _WINDOW_ATTRIBUTES
{
    WNDCLASSEX wcex = { 0 };
    LPCWSTR name = CRC_WND_DEFAULT_NAME;
    unsigned int initialPosX = CRC_WND_DEFAULT_POS_X;
    unsigned int initialPosY = CRC_WND_DEFAULT_POS_Y;
    unsigned int width = CRC_WND_DEFAULT_WIDTH;
    unsigned int height = CRC_WND_DEFAULT_HEIGHT;
    DWORD style = WS_OVERLAPPEDWINDOW;
    HWND hWndParent = NULL;
    HINSTANCE hInstance = nullptr;
} WNDATTR;

class CRC_API Window
{
private:
    HWND hWnd;
    WNDATTR wattr;
    Window(WNDATTR wattr);

public:
    ~Window();

    HRESULT Register();
    HRESULT Create();
    HRESULT Show(int nCmdShow);
    HRESULT Destroy();

    friend class WindowFactory;
};

class CRC_API WindowController
{
    
};

class CRC_API WindowFactory
{
private:
    WindowFactory();
    std::vector<std::shared_ptr<Window>> windows;

public:
    ~WindowFactory();
    static WindowFactory* GetInstance();

    CRC_SLOT CreateWindowCRC(WNDATTR wattr);
    HRESULT DestroyWindowCRC(CRC_SLOT slot);
    HRESULT ShowWindowCRC(CRC_SLOT slot, int nCmdShow);

    static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
};

} // namespace CRC