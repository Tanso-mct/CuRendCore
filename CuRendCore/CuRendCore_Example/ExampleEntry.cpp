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
        // Create window by window attributes.
        std::unique_ptr<CRCWindowAttr> windowAttr = std::make_unique<CRCWindowAttr>();
        windowAttr->wcex_.lpszClassName = L"Main Window";
        windowAttr->wcex_.lpfnWndProc = WindowProc;
        windowAttr->name_ = L"Main Window";
        windowAttr->hInstance = hInstance;

        std::unique_ptr<CRCData> windowData = CRC::CreateWindowData(windowAttr);
        if (!windowData) return CRC::ERROR_CREATE_WINDOW;

        // Add window to window container.
        idMainWindow = CRC::Core()->WindowContainer()->Add(windowData);
    }

    // Scene creation.
    int idMainScene = CRC::INVALID_ID;
    {
        // Create scene by scene attributes.
        std::unique_ptr<CRCSceneAttr> sceneAttr = std::make_unique<CRCSceneAttr>();
        sceneAttr->name_ = "MainScene";

        std::unique_ptr<CRCData> sceneData = CRC::CreateSceneData(sceneAttr);
        if (!sceneData) return CRC::ERROR_CREATE_SCENE;

        // Add scene to scene container.
        idMainScene = CRC::Core()->SceneContainer()->Add(sceneData);
    }

    // Set scene to window.
    HRESULT hr = CRC::Core()->SetSceneToWindow(idMainWindow, idMainScene);
    if (FAILED(hr)) return CRC::ERROR_SET_SCENE_TO_WINDOW;

    CRC::CreateWindowsAsync();
    CRC::CreateScenesAsync();

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