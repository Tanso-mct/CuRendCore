#include "CRC_pch.h"

#include "CuRendCore.h"

#include <Windows.h>

#include "window.h"
#include "scene.h"

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

int APIENTRY WinMain
(
    _In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPSTR lpCmdLine, _In_ int nCmdShow
) {
    // Core initialization.
    CRC::Core()->Initialize();

    // Window data creation.
    int idMainWindow = CRC::ID_INVALID;
    {
        // Create window container.
        std::unique_ptr<ICRCContainer> windowContainer = std::make_unique<CRCWindowContainer>();

        // Create window by window attributes.
        std::unique_ptr<CRCWindowSrc> windowAttr = std::make_unique<CRCWindowSrc>();
        windowAttr->wcex_.lpszClassName = L"Main Window";
        windowAttr->wcex_.lpfnWndProc = WindowProc;
        windowAttr->name_ = L"Main Window";
        windowAttr->hInstance = hInstance;
        std::unique_ptr<ICRCContainable> windowData = CRC::CreateWindowData(std::move(windowAttr));

        // Add window to window container.
        idMainWindow = windowContainer->Add(std::move(windowData));

        // Move window container to core.
        CRC::Core()->SetWindowContainer(std::move(windowContainer));
    }

    // Scene data creation.
    int idMainScene = CRC::ID_INVALID;
    {
        // Create scene container.
        std::unique_ptr<ICRCContainer> sceneContainer = std::make_unique<CRCSceneContainer>();

        // Create scene by scene attributes.
        std::unique_ptr<CRCSceneSrc> sceneAttr = std::make_unique<CRCSceneSrc>();
        sceneAttr->name_ = "MainScene";
        std::unique_ptr<ICRCContainable> sceneData = CRC::CreateSceneData(std::move(sceneAttr));

        // Add scene to scene container.
        idMainScene = sceneContainer->Add(std::move(sceneData));

        // Move scene container to core.
        CRC::Core()->SetSceneContainer(std::move(sceneContainer));
    }

    HRESULT hr = S_OK;

    // Create phase method.
    std::unique_ptr<ICRCPhaseMethod> mainWindowPhaseMethod = std::make_unique<MainWindowPhaseMethod>();
    std::unique_ptr<ICRCPhaseMethod> mainScenePhaseMethod = std::make_unique<MainScenePhaseMethod>();

    // Create window.
    hr = CRC::Core()->CreateWindowCRC(idMainWindow, std::move(mainWindowPhaseMethod));
    if (FAILED(hr)) return CRC::ERROR_CREATE_WINDOW;

    // Show window.
    hr = CRC::Core()->ShowWindowCRC(idMainWindow);
    if (FAILED(hr)) return CRC::ERROR_SHOW_WINDOW;

    // Create scene.
    hr = CRC::Core()->CreateScene(idMainScene, std::move(mainScenePhaseMethod));
    if (FAILED(hr)) return CRC::ERROR_CREATE_SCENE;

    // Set scene to window.
    hr = CRC::Core()->SetSceneToWindow(idMainWindow, idMainScene);
    if (FAILED(hr)) return CRC::ERROR_SET_SCENE_TO_WINDOW;

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