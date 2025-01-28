#include "CRC_pch.h"

#include "CuRendCore.h"

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

int APIENTRY WinMain
(
    _In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPSTR lpCmdLine, _In_ int nCmdShow
) {
    // Core initialization.
    CRC::Core()->Initialize();

    // Window creation.
    int idMainWindow = CRC::INVALID_ID;
    {
        // Create window container.
        std::unique_ptr<CRCContainer> windowContainer = CRC::CreateWindowContainer();

        // Create window by window attributes.
        CRCWindowAttr windowAttr;
        windowAttr.hInstance = hInstance;
        windowAttr.wcex_.lpfnWndProc = WindowProc;

        std::unique_ptr<CRCData> windowData = CRC::CreateCRCWindow(windowAttr);
        if (!windowData) return CRC::ERROR_CREATE_WINDOW;

        HRESULT hr = CRC::ShowCRCWindow(windowData);
        if (FAILED(hr)) return CRC::ERROR_SHOW_WINDOW;

        // Add window to window container.
        idMainWindow = windowContainer->Add(windowData);

        // Move window container to core.
        CRC::Core()->MoveWindowContainer(windowContainer);
    }

    // Scene creation.
    {
        // Create scene container.
        std::unique_ptr<CRCContainer> sceneContainer = CRC::CreateSceneContainer();

        // Move scene container to core.
        CRC::Core()->MoveSceneContainer(sceneContainer);
    }

    CRC::Core()->Run();

    return CRC::Core()->Shutdown();
}

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, msg, wParam, lParam);
    }

    return 0;
}