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
    std::unique_ptr<WindowController> ctrl = nullptr;
} WND_ATTR;


class CRC_API Window
{
private:
    Window(WND_ATTR& wattr);

    HWND hWnd = NULL;
    
    CRC_SLOT thisSlot = CRC_SLOT_INVALID;
    WNDCLASSEX wcex = { 0 };
    LPCWSTR name = CRC_WND_DEFAULT_NAME;
    int initialPosX = CRC_WND_DEFAULT_POS_X;
    int initialPosY = CRC_WND_DEFAULT_POS_Y;
    unsigned int width = CRC_WND_DEFAULT_WIDTH;
    unsigned int height = CRC_WND_DEFAULT_HEIGHT;
    DWORD style = WS_OVERLAPPEDWINDOW;
    HWND hWndParent = NULL;
    HINSTANCE hInstance = nullptr;

    std::unique_ptr<WindowController> ctrl = nullptr;
    std::shared_ptr<Input> input;

public:
    ~Window();

    Window(const Window&) = delete; // Delete copy constructor
    Window& operator=(const Window&) = delete; // Remove copy assignment operator

    Window(Window&&) = delete; // Delete move constructor
    Window& operator=(Window&&) = delete; // Delete move assignment operator

    CRC_SLOT GetSlot() { return thisSlot; }

    friend class WindowFactory;
};

class CRC_API WindowController
{
private:
    std::weak_ptr<Scene> scene;
    std::weak_ptr<Input> input;

protected:
    // Obtain a scene.It is not recommended to use this by storing it in a non-temporary variable.
    std::shared_ptr<Scene> GetScene(){return scene.lock();};

    // Get the scene's weak_ptr unlike GetScene, there is no problem storing it in a non-temporary variable for use.
    std::weak_ptr<Scene> GetSceneWeak(){return scene;};

    // Obtain an input.It is not recommended to use this by storing it in a non-temporary variable.
    std::shared_ptr<Input> GetInput(){return input.lock();};

    // Get the input's weak_ptr unlike GetInput, there is no problem storing it in a non-temporary variable for use.
    std::weak_ptr<Input> GetInputWeak(){return input;};

public:
    WindowController() {CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};
    virtual ~WindowController() {CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};

    WindowController(const WindowController&) = delete; // Delete copy constructor
    WindowController& operator=(const WindowController&) = delete; // Remove copy assignment operator

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
    WindowFactory(){CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};
    std::vector<std::shared_ptr<Window>> windows;
    std::unordered_map<HWND, CRC_SLOT> slots;

    Window* creatingWnd;

public:
    ~WindowFactory();

    CRC_SLOT CreateWindowCRC(WND_ATTR& wattr);
    HRESULT DestroyWindowCRC(CRC_SLOT slot);
    HRESULT ShowWindowCRC(CRC_SLOT slot, int nCmdShow);

    HRESULT SetScene(CRC_SLOT slotWnd, CRC_SLOT slotScene);

    static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

    friend class CuRendCore;
};

} // namespace CRC