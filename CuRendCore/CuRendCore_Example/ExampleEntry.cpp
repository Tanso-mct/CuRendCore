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
        windowAttr.wcex_.lpszClassName = L"Main Window";
        windowAttr.wcex_.lpfnWndProc = WindowProc;
        windowAttr.name_ = L"Main Window";
        windowAttr.hInstance = hInstance;

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
    int idMainScene = CRC::INVALID_ID;
    {
        // Create scene container.
        std::unique_ptr<CRCContainer> sceneContainer = CRC::CreateSceneContainer();

        // Create scene by scene attributes.
        CRCSceneAttr sceneAttr;
        sceneAttr.name_ = "MainScene";

        std::unique_ptr<CRCData> sceneData = CRC::CreateCRCScene(sceneAttr);
        if (!sceneData) return CRC::ERROR_CREATE_SCENE;

        // Add scene to scene container.
        idMainScene = sceneContainer->Add(sceneData);

        // Move scene container to core.
        CRC::Core()->MoveSceneContainer(sceneContainer);
    }

    // Main loop.
    MSG msg = {};
    while (msg.message != WM_QUIT)
    {
        // Process any messages in the queue.
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

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