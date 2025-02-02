#include "CRC_pch.h"

#include "CuRendCore.h"
#pragma comment(lib, "CuRendCore.lib")

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

int APIENTRY WinMain
(
    _In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPSTR lpCmdLine, _In_ int nCmdShow
) {
    // Core initialization.
    CRC::Core()->Initialize();

    // Window data creation.
    int idMainWindow = CRC::INVALID_ID;
    {
        // Create window container.
        std::unique_ptr<CRCContainer> windowContainer = std::make_unique<CRCWindowContainer>();

        // Create window by window attributes.
        std::unique_ptr<CRCWindowAttr> windowAttr = std::make_unique<CRCWindowAttr>();
        windowAttr->wcex_.lpszClassName = L"Main Window";
        windowAttr->wcex_.lpfnWndProc = WindowProc;
        windowAttr->name_ = L"Main Window";
        windowAttr->hInstance = hInstance;
        std::unique_ptr<CRCData> windowData = CRC::CreateWindowData(std::move(windowAttr));

        // Add window to window container.
        idMainWindow = windowContainer->Add(windowData);

        // Move window container to core.
        CRC::Core()->SetWindowContainer(std::move(windowContainer));
    }

    // Scene data creation.
    int idMainScene = CRC::INVALID_ID;
    {
        // Create scene container.
        std::unique_ptr<CRCContainer> sceneContainer = std::make_unique<CRCSceneContainer>();

        // Create scene by scene attributes.
        std::unique_ptr<CRCSceneAttr> sceneAttr = std::make_unique<CRCSceneAttr>();
        sceneAttr->name_ = "MainScene";
        std::unique_ptr<CRCData> sceneData = CRC::CreateSceneData(std::move(sceneAttr));

        // Add scene to scene container.
        idMainScene = sceneContainer->Add(sceneData);

        // Move scene container to core.
        CRC::Core()->SetSceneContainer(std::move(sceneContainer));
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