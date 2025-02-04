#include "CRC_pch.h"

#include "CuRendCore.h"

#include <Windows.h>

#include "window.h"
#include "scene.h"

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

int main() 
{
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
        windowAttr->hInstance =  GetModuleHandle(NULL);
        std::unique_ptr<ICRCContainable> windowData = CRC::CreateWindowData(std::move(windowAttr));

        // Add window to window container.
        idMainWindow = windowContainer->Add(std::move(windowData));

        // Move window container to core.
        CRC::Core()->SetWindowContainer(std::move(windowContainer));
    }

    // Window phase method creation.
    int idMainWindowPM = CRC::ID_INVALID;
    {
        // Create window phase method container.
        std::unique_ptr<ICRCContainer> windowPMContainer = std::make_unique<CRCPMContainer>();

        // Create window phase method by window phase method attributes.
        std::unique_ptr<ICRCPhaseMethod> windowPMData = std::make_unique<MainWindowPhaseMethod>();

        // Add window phase method to window phase method container.
        idMainWindowPM = windowPMContainer->Add(std::move(windowPMData));

        // Move window phase method container to core.
        CRC::Core()->SetWindowPMContainer(std::move(windowPMContainer));
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

    // Scene phase method creation.
    int idMainScenePM = CRC::ID_INVALID;
    {
        // Create scene phase method container.
        std::unique_ptr<ICRCContainer> scenePMContainer = std::make_unique<CRCPMContainer>();

        // Create scene phase method by scene phase method attributes.
        std::unique_ptr<ICRCPhaseMethod> scenePMData = std::make_unique<MainScenePhaseMethod>();

        // Add scene phase method to scene phase method container.
        idMainScenePM = scenePMContainer->Add(std::move(scenePMData));

        // Move scene phase method container to core.
        CRC::Core()->SetScenePMContainer(std::move(scenePMContainer));
    }

    HRESULT hr = S_OK;

    // Create window.
    hr = CRC::Core()->CreateWindowCRC(idMainWindow, idMainWindowPM);
    if (FAILED(hr)) return CRC::ERROR_CREATE_WINDOW;

    // Show window.
    hr = CRC::Core()->ShowWindowCRC(idMainWindow);
    if (FAILED(hr)) return CRC::ERROR_SHOW_WINDOW;

    // Create scene.
    hr = CRC::Core()->CreateScene(idMainScene, idMainScenePM);
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
    CRC::Core()->HandleWindowProc(hWnd, msg, wParam, lParam);

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