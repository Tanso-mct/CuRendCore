#include "Window.h"
#include "CuRendCore.h"

namespace CRC 
{

Window::Window(WNDATTR wattr)
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    this->wattr = wattr;
    this->ctrl = wattr.ctrl;
    input = std::shared_ptr<Input>(new Input());
}

Window::~Window()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    if (ctrl != nullptr) ctrl.reset();
    if (input != nullptr) input.reset();
}

WindowController::WindowController()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");
}

WindowController::~WindowController()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");
}

WindowFactory::~WindowFactory()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");
    
    for (auto& window : windows)
    {
        window.reset();
    }
    windows.clear();
    slots.clear();
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
    if (windows[slot] == nullptr) return E_FAIL;

    HRESULT hr = !DestroyWindow(windows[slot]->hWnd);
    if (SUCCEEDED(hr))
    {
        windows[slot].reset();
    }

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
    if (windows[slot] == nullptr) return E_FAIL;

    if (windows[slot]->ctrl->GetSceneCtrl() != nullptr)
    {
        // Since the Scene controller is switched, it means that the scene is switched.
        windows[slot]->ctrl->GetSceneCtrl()->Finish();
    }

    windows[slot]->ctrl->sceneCtrl = sceneCtrl;
    windows[slot]->ctrl->GetSceneCtrl()->input = windows[slot]->ctrl->input;
    
    return S_OK;
}

LRESULT CALLBACK WindowFactory::WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (wParam == SC_MINIMIZE)
    {
        WindowFactory* wf = CuRendCore::GetInstance()->windowFc;
        wf->windows[wf->slots[hWnd]]->ctrl->OnMinimize(hWnd, msg, wParam, lParam);
    }

    switch (msg){
        case WM_SIZE:
        {
            WindowFactory* wf = CuRendCore::GetInstance()->windowFc;
            if (wParam == SIZE_MINIMIZED) wf->windows[wf->slots[hWnd]]->ctrl->OnMinimize(hWnd, msg, wParam, lParam);
            else if (wParam == SIZE_MAXIMIZED) wf->windows[wf->slots[hWnd]]->ctrl->OnMaximize(hWnd, msg, wParam, lParam);
            else if (wParam == SIZE_RESTORED) wf->windows[wf->slots[hWnd]]->ctrl->OnRestored(hWnd, msg, wParam, lParam);
        }
            break;

        case WM_SETFOCUS:
        {
            WindowFactory* wf = CuRendCore::GetInstance()->windowFc;
            wf->windows[wf->slots[hWnd]]->ctrl->OnSetFocus(hWnd, msg, wParam, lParam);
        }
            break;

        case WM_KILLFOCUS:
        {
            WindowFactory* wf = CuRendCore::GetInstance()->windowFc;
            wf->windows[wf->slots[hWnd]]->ctrl->OnKillFocus(hWnd, msg, wParam, lParam);
        }
            break;

        case WM_CREATE:
        {
            WindowFactory* wf = CuRendCore::GetInstance()->windowFc;
            wf->creatingWnd->thisSlot = wf->windows.size();
            wf->slots[wf->creatingWnd->hWnd] = wf->windows.size();

            wf->creatingWnd->ctrl->input = wf->creatingWnd->input;
            wf->creatingWnd->hWnd = hWnd;
            wf->creatingWnd->ctrl->OnCreate(hWnd, msg, wParam, lParam);

            wf->windows.push_back(wf->creatingWnd);
        }
            break;

        case WM_PAINT:
        {
            WindowFactory* wf = CuRendCore::GetInstance()->windowFc;
            if (wf->windows[wf->slots[hWnd]]->ctrl->GetSceneCtrl() != nullptr)
            {
                wf->windows[wf->slots[hWnd]]->ctrl->OnPaint(hWnd, msg, wParam, lParam);
                wf->windows[wf->slots[hWnd]]->ctrl->GetSceneCtrl()->OnPaint();
                wf->windows[wf->slots[hWnd]]->ctrl->GetInput()->Update();
            }
        }
            break;

        case WM_MOVE:
        {
            WindowFactory* wf = CuRendCore::GetInstance()->windowFc;
            wf->windows[wf->slots[hWnd]]->ctrl->OnMove(hWnd, msg, wParam, lParam);
        }
            break;

        case WM_CLOSE:
        {
            WindowFactory* wf = CuRendCore::GetInstance()->windowFc;
            wf->windows[wf->slots[hWnd]]->ctrl->OnClose(hWnd, msg, wParam, lParam);
            wf->windows[wf->slots[hWnd]]->ctrl->GetSceneCtrl()->Finish();
        }
            return DefWindowProc(hWnd, msg, wParam, lParam);

        case WM_DESTROY:
        {
            WindowFactory* wf = CuRendCore::GetInstance()->windowFc;
            wf->windows[wf->slots[hWnd]]->ctrl->OnDestroy(hWnd, msg, wParam, lParam);   
        }
            break;

        case WM_SYSKEYDOWN:
        case WM_KEYDOWN:
        {
            WindowFactory* wf = CuRendCore::GetInstance()->windowFc;
            wf->windows[wf->slots[hWnd]]->ctrl->GetInput()->ProcessKeyDown(hWnd, msg, wParam, lParam);
            wf->windows[wf->slots[hWnd]]->ctrl->OnKeyDown(hWnd, msg, wParam, lParam);
        }
            break;

        case WM_SYSKEYUP:
        case WM_KEYUP:
        {
            WindowFactory* wf = CuRendCore::GetInstance()->windowFc;
            wf->windows[wf->slots[hWnd]]->ctrl->GetInput()->ProcessKeyUp(hWnd, msg, wParam, lParam);
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
            WindowFactory* wf = CuRendCore::GetInstance()->windowFc;
            wf->windows[wf->slots[hWnd]]->ctrl->GetInput()->ProcessMouse(hWnd, msg, wParam, lParam);
            wf->windows[wf->slots[hWnd]]->ctrl->OnMouse(hWnd, msg, wParam, lParam);
        }
            break;
            
        default:
            return DefWindowProc(hWnd, msg, wParam, lParam);
    }

    return 0;
}

} // namespace CRC
