#include "Window.h"

namespace CRC 
{

Window::Window(WNDATTR wattr)
{
    this->wattr = wattr;
}

Window::~Window()
{
}

HRESULT Window::Register()
{
    if (!RegisterClassEx(&wattr.wcex)) return E_FAIL;
    return S_OK;
}

HRESULT Window::Create()
{
    hWnd = CreateWindow(
        wattr.wcex.lpszClassName,
        wattr.name,
        wattr.style,
        wattr.initialPosX, wattr.initialPosY,
        wattr.width, wattr.height,
        wattr.hWndParent,
        NULL,
        wattr.hInstance,
        NULL
    );

    if (!hWnd) return E_FAIL;
    return S_OK;
}

HRESULT Window::Show(int nCmdShow)
{
    HRESULT hr = S_OK;
    hr = ShowWindow(hWnd, nCmdShow);
    if (FAILED(hr)) return hr;

    hr = UpdateWindow(hWnd);
    if (FAILED(hr)) return hr;

    return S_OK;
}

HRESULT Window::Destroy()
{
    if (!DestroyWindow(hWnd)) return E_FAIL;
    return S_OK;
}

WindowFactory::WindowFactory()
{
}

WindowFactory::~WindowFactory()
{
}

WindowFactory *WindowFactory::GetInstance()
{
    // Implementation of the Singleton pattern.
    static WindowFactory* instance = nullptr;

    if (instance == nullptr)
    {
        instance = new WindowFactory();
    }

    return instance;
}

CRC_SLOT WindowFactory::CreateWindowCRC(WNDATTR wattr)
{
    if (wattr.wcex.cbSize == 0) return CRC_SLOT_INVALID; // The window class is not initialized.
    wattr.wcex.lpfnWndProc = WindowProc;

    std::shared_ptr<Window> newWnd = std::shared_ptr<Window>(new Window(wattr));
    newWnd->Register();
    newWnd->Create();

    windows.push_back(newWnd);

    return (CRC_SLOT)(windows.size() - 1);
}

HRESULT WindowFactory::DestroyWindowCRC(CRC_SLOT slot)
{
    if (slot >= windows.size()) return E_FAIL;

    HRESULT hr = windows[slot]->Destroy();
    if (SUCCEEDED(hr)) windows.erase(windows.begin() + slot);

    return hr;
}

HRESULT WindowFactory::ShowWindowCRC(CRC_SLOT slot, int nCmdShow)
{
    if (slot >= windows.size()) return E_FAIL;

    HRESULT hr = windows[slot]->Show(nCmdShow);
    return hr;
}

LRESULT CALLBACK WindowFactory::WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (wParam == SC_MINIMIZE)
    {
        OutputDebugStringA("Minimized\n");
    }

    switch (msg){
        case WM_GETMINMAXINFO:
            OutputDebugStringA("WM_GETMINMAXINFO\n");
            break;

        case WM_SETFOCUS:
            OutputDebugStringA("WM_SETFOCUS\n");
            break;

        case WM_KILLFOCUS:
            OutputDebugStringA("WM_KILLFOCUS\n");
            break;

        case WM_CREATE:
            OutputDebugStringA("WM_CREATE\n");
            break;

        case WM_PAINT:

            break;

        case WM_MOVE:
            OutputDebugStringA("WM_MOVE\n");
            break;

        case WM_CLOSE:
            OutputDebugStringA("WM_CLOSE\n");
            return DefWindowProc(hWnd, msg, wParam, lParam);

        case WM_DESTROY:
            OutputDebugStringA("WM_DESTROY\n");
            PostQuitMessage(0);
            break;

        case WM_SYSKEYDOWN:
        case WM_KEYDOWN:
            OutputDebugStringA("WM_KEYDOWN\n");
            break;

        case WM_SYSKEYUP:
        case WM_KEYUP:
            OutputDebugStringA("WM_KEYUP\n");
            break;

        case WM_LBUTTONDOWN:
        case WM_LBUTTONUP:
        case WM_LBUTTONDBLCLK:
        case WM_RBUTTONDOWN:
        case WM_RBUTTONUP:
        case WM_RBUTTONDBLCLK:
        case WM_MBUTTONDOWN:
        case WM_MBUTTONUP:
        case WM_MOUSEWHEEL:
        case WM_MOUSEMOVE:
            break;
            
        default:
            return DefWindowProc(hWnd, msg, wParam, lParam);
    }

    return 0;
}

} // namespace CRC
