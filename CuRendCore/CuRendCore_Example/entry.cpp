#include "CRC_pch.h"

#include "CuRendCore.h"

#include <Windows.h>

#include "window.h"
#include "scene.h"

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

int main() 
{
    /**************************************************************************************************************** */
    // Core initialization.
    /**************************************************************************************************************** */
    CRC::Core()->Initialize();

    /**************************************************************************************************************** */
    // Window data creation.
    /**************************************************************************************************************** */
    int idMainWindow = CRC::ID_INVALID;
    int idWindowContainer = CRC::ID_INVALID;
    {
        // Create window container.
        std::unique_ptr<ICRCContainer> container = std::make_unique<CRCContainer>();

        // Create window by window attributes.
        std::unique_ptr<CRCWindowSrc> windowAttr = std::make_unique<CRCWindowSrc>();
        windowAttr->wcex_.lpszClassName = L"Main Window";
        windowAttr->wcex_.lpfnWndProc = WindowProc;
        windowAttr->name_ = L"Main Window";
        windowAttr->hInstance =  GetModuleHandle(NULL);
        std::unique_ptr<ICRCContainable> windowData = CRC::CreateWindowData(std::move(windowAttr));

        // Add window to window container.
        idMainWindow = container->Add(std::move(windowData));

        // Move window container to core.
        idWindowContainer = CRC::Core()->AddContainer(std::move(container));
    }
    if (idMainWindow == CRC::ID_INVALID) return CRC::ERROR_CREATE_WINDOW;
    if (idWindowContainer == CRC::ID_INVALID) return CRC::ERROR_CREATE_CONTAINER;

    /**************************************************************************************************************** */
    // Window phase method creation.
    /**************************************************************************************************************** */
    int idMainWindowPM = CRC::ID_INVALID;
    int idWindowPMContainer = CRC::ID_INVALID;
    {
        // Create window phase method container.
        std::unique_ptr<ICRCContainer> container = std::make_unique<CRCContainer>();

        // Create window phase method by client's window phase method class.
        std::unique_ptr<ICRCPhaseMethod> windowPMData = std::make_unique<MainWindowPhaseMethod>();

        // Add window phase method to window phase method container.
        idMainWindowPM = container->Add(std::move(windowPMData));

        // Move window phase method container to core.
        idWindowPMContainer = CRC::Core()->AddContainer(std::move(container));
    }
    if (idMainWindowPM == CRC::ID_INVALID) return CRC::ERROR_CREATE_PM;
    if (idWindowPMContainer == CRC::ID_INVALID) return CRC::ERROR_CREATE_CONTAINER;

    /**************************************************************************************************************** */
    // Scene data creation.
    /**************************************************************************************************************** */
    int idMainScene = CRC::ID_INVALID;
    int idSceneContainer = CRC::ID_INVALID;
    {
        // Create scene container.
        std::unique_ptr<ICRCContainer> container = std::make_unique<CRCContainer>();

        // Create scene by scene attributes.
        std::unique_ptr<CRCSceneSrc> sceneAttr = std::make_unique<CRCSceneSrc>();
        sceneAttr->name_ = "MainScene";
        std::unique_ptr<ICRCContainable> sceneData = CRC::CreateSceneData(std::move(sceneAttr));

        // Add scene to scene container.
        idMainScene = container->Add(std::move(sceneData));

        // Move scene container to core.
        idSceneContainer = CRC::Core()->AddContainer(std::move(container));
    }
    if (idMainScene == CRC::ID_INVALID) return CRC::ERROR_CREATE_CONTAINER;
    if (idSceneContainer == CRC::ID_INVALID) return CRC::ERROR_CREATE_CONTAINER;

    /**************************************************************************************************************** */
    // Scene phase method creation.
    /**************************************************************************************************************** */
    int idMainScenePM = CRC::ID_INVALID;
    int idScenePMContainer = CRC::ID_INVALID;
    {
        // Create scene phase method container.
        std::unique_ptr<ICRCContainer> container = std::make_unique<CRCContainer>();

        // Create scene phase method by client's scene phase method class.
        std::unique_ptr<ICRCPhaseMethod> scenePMData = std::make_unique<MainScenePhaseMethod>();

        // Add scene phase method to scene phase method container.
        idMainScenePM = container->Add(std::move(scenePMData));

        // Move scene phase method container to core.
        idScenePMContainer = CRC::Core()->AddContainer(std::move(container));
    }
    if (idMainScenePM == CRC::ID_INVALID) return CRC::ERROR_CREATE_PM;
    if (idScenePMContainer == CRC::ID_INVALID) return CRC::ERROR_CREATE_CONTAINER;

    /**************************************************************************************************************** */
    // Create window, show window, create scene, set scene to window.
    /**************************************************************************************************************** */
    HRESULT hr = S_OK;

    // Create window.
    hr = CRC::Core()->CreateWindowCRC(idMainWindow, idWindowContainer, idMainWindowPM, idWindowPMContainer);
    if (FAILED(hr)) return CRC::ERROR_CREATE_WINDOW;

    // Show window.
    hr = CRC::Core()->ShowWindowCRC(idMainWindow, idWindowContainer);
    if (FAILED(hr)) return CRC::ERROR_SHOW_WINDOW;

    // Create scene.
    hr = CRC::Core()->CreateScene(idMainScene, idSceneContainer, idMainScenePM, idScenePMContainer);
    if (FAILED(hr)) return CRC::ERROR_CREATE_SCENE;

    // Set scene to window.
    hr = CRC::Core()->SetSceneToWindow(idMainWindow, idWindowContainer, idMainScene, idSceneContainer);
    if (FAILED(hr)) return CRC::ERROR_SET_SCENE_TO_WINDOW;

    /**************************************************************************************************************** */
    // Main loop.
    /**************************************************************************************************************** */
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

    /**************************************************************************************************************** */
    // Core shutdown.
    /**************************************************************************************************************** */
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