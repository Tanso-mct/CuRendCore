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
    if (ctrl != nullptr) ctrl.reset();
}

WindowController::WindowController()
{
    input = Input::GetInstance();
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

void WindowFactory::ReleaseInstance()
{
    WindowFactory* instance = GetInstance();
    if (instance != nullptr)
    {
        delete instance;
        instance = nullptr;
    }
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
    creatingWnd = nullptr;
    
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

HRESULT WindowFactory::SetSceneCtrl(CRC_SLOT slot, std::shared_ptr<SceneController> sceneCtrl)
{
    if (slot >= windows.size()) return E_FAIL;

    windows[slot]->ctrl->sceneCtrl = sceneCtrl;
    return S_OK;
}

LRESULT CALLBACK WindowFactory::WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (wParam == SC_MINIMIZE)
    {
        WindowFactory* wf = WindowFactory::GetInstance();
        wf->windows[wf->slots[hWnd]]->ctrl->OnMinimize(hWnd, msg, wParam, lParam);
    }

    switch (msg){
        case WM_SIZE:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            if (wParam == SIZE_MINIMIZED) wf->windows[wf->slots[hWnd]]->ctrl->OnMinimize(hWnd, msg, wParam, lParam);
            else if (wParam == SIZE_MAXIMIZED) wf->windows[wf->slots[hWnd]]->ctrl->OnMaximize(hWnd, msg, wParam, lParam);
            else if (wParam == SIZE_RESTORED) wf->windows[wf->slots[hWnd]]->ctrl->OnRestored(hWnd, msg, wParam, lParam);
        }
            break;

        case WM_SETFOCUS:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->windows[wf->slots[hWnd]]->ctrl->OnSetFocus(hWnd, msg, wParam, lParam);
        }
            break;

        case WM_KILLFOCUS:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->windows[wf->slots[hWnd]]->ctrl->OnKillFocus(hWnd, msg, wParam, lParam);
        }
            break;

        case WM_CREATE:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->creatingWnd->ctrl->OnCreate(hWnd, msg, wParam, lParam);
            wf->creatingWnd->hWnd = hWnd;

            wf->slots[wf->creatingWnd->hWnd] = wf->windows.size();
            wf->windows.push_back(wf->creatingWnd);
        }
            break;

        case WM_PAINT:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->windows[wf->slots[hWnd]]->ctrl->OnPaint(hWnd, msg, wParam, lParam);
            wf->windows[wf->slots[hWnd]]->ctrl->input->Update();
        }
            break;

        case WM_MOVE:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->windows[wf->slots[hWnd]]->ctrl->OnMove(hWnd, msg, wParam, lParam);
        }
            break;

        case WM_CLOSE:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->windows[wf->slots[hWnd]]->ctrl->OnClose(hWnd, msg, wParam, lParam);
        }
            return DefWindowProc(hWnd, msg, wParam, lParam);

        case WM_DESTROY:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->windows[wf->slots[hWnd]]->ctrl->OnDestroy(hWnd, msg, wParam, lParam);   
        }
            break;

        case WM_SYSKEYDOWN:
        case WM_KEYDOWN:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->windows[wf->slots[hWnd]]->ctrl->input->ProcessKeyDown(hWnd, msg, wParam, lParam);
            wf->windows[wf->slots[hWnd]]->ctrl->OnKeyDown(hWnd, msg, wParam, lParam);
        }
            break;

        case WM_SYSKEYUP:
        case WM_KEYUP:
        {
            WindowFactory* wf = WindowFactory::GetInstance();
            wf->windows[wf->slots[hWnd]]->ctrl->input->ProcessKeyUp(hWnd, msg, wParam, lParam);
            wf->windows[wf->slots[hWnd]]->ctrl->OnKeyUp(hWnd, msg, wParam, lParam);
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
            wf->windows[wf->slots[hWnd]]->ctrl->input->ProcessMouse(hWnd, msg, wParam, lParam);
            wf->windows[wf->slots[hWnd]]->ctrl->OnMouse(hWnd, msg, wParam, lParam);
        }
            break;
            
        default:
            return DefWindowProc(hWnd, msg, wParam, lParam);
    }

    return 0;
}

} // namespace CRC
