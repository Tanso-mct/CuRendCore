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
        std::unique_ptr<ICRCContainable> windowData = CRC::CreateWindowAttr(std::move(windowAttr));

        // Add window to window container.
        idMainWindow = container->Add(std::move(windowData));

        // Move window container to core.
        idWindowContainer = CRC::Core()->containerSet_->Add(std::move(container));
    }
    if (idMainWindow == CRC::ID_INVALID) return CRC::ERROR_CREATE_WINDOW;
    if (idWindowContainer == CRC::ID_INVALID) return CRC::ERROR_CREATE_CONTAINER;

    std::unique_ptr<ICRCContainer>& mainWindowContainer = CRC::Core()->containerSet_->Get(idWindowContainer);
    std::unique_ptr<ICRCContainable>& mainWindowAttr = mainWindowContainer->Get(idMainWindow);

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
        std::unique_ptr<ICRCContainable> sceneData = CRC::CreateSceneAttr(std::move(sceneAttr));

        // Add scene to scene container.
        idMainScene = container->Add(std::move(sceneData));

        // Move scene container to core.
        idSceneContainer = CRC::Core()->containerSet_->Add(std::move(container));
    }
    if (idMainScene == CRC::ID_INVALID) return CRC::ERROR_CREATE_CONTAINER;
    if (idSceneContainer == CRC::ID_INVALID) return CRC::ERROR_CREATE_CONTAINER;

    std::unique_ptr<ICRCContainer>& mainSceneContainer = CRC::Core()->containerSet_->Get(idSceneContainer);
    std::unique_ptr<ICRCContainable>& mainSceneAttr = mainSceneContainer->Get(idMainScene);

    /**************************************************************************************************************** */
    // Phase method creation.
    /**************************************************************************************************************** */
    // Create window phase method by client's window phase method class.
    std::unique_ptr<ICRCPhaseMethod> mainWindowPM = std::make_unique<MainWindowPhaseMethod>();

    // Create scene phase method by client's scene phase method class.
    std::unique_ptr<ICRCPhaseMethod> mainScenePM = std::make_unique<MainScenePhaseMethod>();

    /**************************************************************************************************************** */
    // Create window, show window, create scene, set scene to window.
    /**************************************************************************************************************** */
    HRESULT hr = S_OK;

    // Create window.
    hr = CRC::CreateWindowCRC(mainWindowAttr);
    if (FAILED(hr)) return CRC::ERROR_CREATE_WINDOW;

    // Create scene.
    hr = CRC::CreateScene(mainSceneAttr);
    if (FAILED(hr)) return CRC::ERROR_CREATE_SCENE;

    /**************************************************************************************************************** */
    // Set phase method to window.
    /**************************************************************************************************************** */

    CRC::Core()->pmCaller_->Add
    (
        std::move(mainWindowPM), // Phase method.
        CRC::PtrAs<CRCWindowAttr>(mainWindowAttr.get()), // Attribute.
        CRC::PtrAs<CRCWindowAttr>(mainWindowAttr.get())->hWnd_ // Key.
    );
    if (FAILED(hr)) return CRC::ERROR_ADD_PM;

    CRC::Core()->pmCaller_->Add
    (
        std::move(mainScenePM), // Phase method.
        CRC::PtrAs<CRCSceneAttr>(mainSceneAttr.get()), // Attribute.
        CRC::PtrAs<CRCWindowAttr>(mainWindowAttr.get())->hWnd_ // Key.
    );
    if (FAILED(hr)) return CRC::ERROR_ADD_PM;

    /**************************************************************************************************************** */
    // Main loop.
    /**************************************************************************************************************** */

    // Show window.
    hr = CRC::ShowWindowCRC(mainWindowAttr);
    if (FAILED(hr)) return CRC::ERROR_SHOW_WINDOW;

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

    case WM_PAINT:
        // WM_PAINT is always processed. If it is not processed, the frame update method of CRC is not called.
        break;

    default:
        return DefWindowProc(hWnd, msg, wParam, lParam);
    }

    return 0;
}