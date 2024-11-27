#include "Window.h"

namespace CRC 
{

Window::Window(WNDATTR wattr)
{
    this->wattr = wattr;
    this->ctrl = wattr.ctrl;
}

Window::~Window()
{
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

    if (instance == nullptr) instance = new WindowFactory();

    return instance;
}

CRC_SLOT WindowFactory::CreateWindowCRC(WNDATTR wattr)
{
    if (wattr.wcex.cbSize == 0) return CRC_SLOT_INVALID; // The window class is not initialized.
    wattr.wcex.lpfnWndProc = WindowProc;

    std::shared_ptr<Window> newWnd = std::shared_ptr<Window>(new Window(wattr));

    // Resiter the window
    if (!RegisterClassEx(&newWnd->wattr.wcex)) return E_FAIL;

    // Create the window
    creatingWnd = newWnd;
    newWnd->hWnd = CreateWindow(
        newWnd->wattr.wcex.lpszClassName,
        newWnd->wattr.name,
        newWnd->wattr.style,
        newWnd->wattr.initialPosX, newWnd->wattr.initialPosY,
        newWnd->wattr.width, newWnd->wattr.height,
        newWnd->wattr.hWndParent,
        NULL,
        newWnd->wattr.hInstance,
        NULL
    );
    if (!newWnd->hWnd) return E_FAIL;

    return (CRC_SLOT)(windows.size() - 1);
}

HRESULT WindowFactory::DestroyWindowCRC(CRC_SLOT slot)
{
    if (slot >= windows.size()) return E_FAIL;

    HRESULT hr = !DestroyWindow(windows[slot]->hWnd);
    if (SUCCEEDED(hr)) windows.erase(windows.begin() + slot);

    return hr;
}

HRESULT WindowFactory::ShowWindowCRC(CRC_SLOT slot, int nCmdShow)
{
    if (slot >= windows.size()) return E_FAIL;

    HRESULT hr = S_OK;
    hr = ShowWindow(windows[slot]->hWnd, nCmdShow);
    if (FAILED(hr)) return hr;

    hr = UpdateWindow(windows[slot]->hWnd);
    if (FAILED(hr)) return hr;

    return hr;
}

LRESULT CALLBACK WindowFactory::WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (wParam == SC_MINIMIZE)
    {
        WindowFactory* wf = WindowFactory::GetInstance();
        wf->windows[wf->slots[hWnd]]->ctrl->OnMinimize();
    }

    switch (msg){
        case WM_SIZE:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            if (wParam == SIZE_MINIMIZED) wf->windows[wf->slots[hWnd]]->ctrl->OnMinimize();
            else if (wParam == SIZE_MAXIMIZED) wf->windows[wf->slots[hWnd]]->ctrl->OnMaximize();
            else if (wParam == SIZE_RESTORED) wf->windows[wf->slots[hWnd]]->ctrl->OnRestored();
        }
            break;

        case WM_SETFOCUS:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->windows[wf->slots[hWnd]]->ctrl->OnSetFocus();
        }
            break;

        case WM_KILLFOCUS:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->windows[wf->slots[hWnd]]->ctrl->OnKillFocus();
        }
            break;

        case WM_CREATE:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->creatingWnd->ctrl->OnCreate();
            wf->creatingWnd->hWnd = hWnd;

            wf->slots[wf->creatingWnd->hWnd] = wf->windows.size();
            wf->windows.push_back(wf->creatingWnd);
        }
            break;

        case WM_PAINT:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->windows[wf->slots[hWnd]]->ctrl->OnPaint();
        }
            break;

        case WM_MOVE:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->windows[wf->slots[hWnd]]->ctrl->OnMove();
        }
            break;

        case WM_CLOSE:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->windows[wf->slots[hWnd]]->ctrl->OnClose();
        }
            return DefWindowProc(hWnd, msg, wParam, lParam);

        case WM_DESTROY:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->windows[wf->slots[hWnd]]->ctrl->OnDestroy();   
        }
            break;

        case WM_SYSKEYDOWN:
        case WM_KEYDOWN:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->windows[wf->slots[hWnd]]->ctrl->OnKeyDown();
        }
            break;

        case WM_SYSKEYUP:
        case WM_KEYUP:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->windows[wf->slots[hWnd]]->ctrl->OnKeyUp();
        }
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
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->windows[wf->slots[hWnd]]->ctrl->OnMouse();
        }
            break;
            
        default:
            return DefWindowProc(hWnd, msg, wParam, lParam);
    }

    return 0;
}

} // namespace CRC
