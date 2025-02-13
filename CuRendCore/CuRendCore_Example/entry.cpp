﻿#include "CRC_pch.h"

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
    std::unique_ptr<ICRCWinMsgListener> mainWindowL = std::make_unique<MainWindowListener>();

    // Create scene phase method by client's scene phase method class.
    std::unique_ptr<ICRCWinMsgListener> mainSceneL = std::make_unique<MainSceneListener>();

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
    // Set Windows Message Listener to window.
    /**************************************************************************************************************** */

    /**
     * when processing the window's message.
     * Also, the order in which Listener are called is the order in which they are added to winMsgCaller_.
     */

    CRC::Core()->winMsgCaller_->Add // Window Listener.
    (
        CRC::PtrAs<CRCWindowAttr>(mainWindowAttr.get())->hWnd_, // Key.
        std::move(mainWindowL), // Listener.
        CRC::PtrAs<CRCWindowAttr>(mainWindowAttr.get()) // Attribute.
    );
    if (FAILED(hr)) return CRC::ERROR_ADD_LISTENER;

    CRC::Core()->winMsgCaller_->Add // Scene Listener.
    (
        CRC::PtrAs<CRCWindowAttr>(mainWindowAttr.get())->hWnd_, // Key.
        std::move(mainSceneL), // Listener.
        CRC::PtrAs<CRCWindowAttr>(mainWindowAttr.get()) // Attribute.
    );
    if (FAILED(hr)) return CRC::ERROR_ADD_LISTENER;

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
    case WM_PAINT:
        CRC::Core()->FrameUpdate(hWnd, msg, wParam, lParam);
        break;

    default:
        return DefWindowProc(hWnd, msg, wParam, lParam);
    }

    return 0;
}