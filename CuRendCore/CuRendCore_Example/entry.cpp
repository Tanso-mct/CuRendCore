#include "CuRendCore/include/CuRendCore.h"

#include <Windows.h>

#include "window.h"
#include "scene.h"

static std::unique_ptr<WACore::IContainer> gContainer = nullptr; // Attributes container.
static WACore::WPEventCaller gWPEventCaller; // Window message event caller.

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

int main() 
{
    /******************************************************************************************************************
     * Attributes container creation.
     *****************************************************************************************************************/
    gContainer = std::make_unique<WACore::Container>();
    if (!gContainer) return CRC::ERROR_CREATE_CONTAINER;

    /******************************************************************************************************************
     * Window attributes creation.
     *****************************************************************************************************************/
    int idMainWindowAttr = CRC::ID_INVALID;
    {
        // Create window factory.
        CRCWindowFactory windowFactory;

        // Create window attributes.
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"Main Window";
        desc.wcex_.lpfnWndProc = WindowProc;
        desc.name_ = L"Main Window";
        desc.hInstance = GetModuleHandle(NULL);
        std::unique_ptr<WACore::IContainable> windowAttr = windowFactory.Create(desc);

        // Add window attribute to container.
        idMainWindowAttr = gContainer->Add(std::move(windowAttr));
    }
    if (idMainWindowAttr == CRC::ID_INVALID) return CRC::ERROR_ADD_TO_CONTAINER;

    /******************************************************************************************************************
     * Scene attributes creation.
     *****************************************************************************************************************/
    int idMainSceneAttr = CRC::ID_INVALID;
    {
        // Create scene factory.
        CRCSceneFactory sceneFactory;

        // Create scene attributes.
        CRC_SCENE_DESC desc = {};
        desc.name_ = "MainScene";
        std::unique_ptr<WACore::IContainable> sceneAttr = sceneFactory.Create(desc);

        // Add scene to scene container.
        idMainSceneAttr = gContainer->Add(std::move(sceneAttr));
    }
    if (idMainSceneAttr == CRC::ID_INVALID) return CRC::ERROR_ADD_TO_CONTAINER;

    /******************************************************************************************************************
     * User Input attributes creation.
     *****************************************************************************************************************/
    int idUserInputAttr = CRC::ID_INVALID;
    {
        // Create user input by user input attributes.
        std::unique_ptr<WACore::IContainable> userInputAttr = std::make_unique<CRCUserInputAttr>();

        // Add user input to user input container.
        idUserInputAttr = gContainer->Add(std::move(userInputAttr));
    }
    if (idUserInputAttr == CRC::ID_INVALID) return CRC::ERROR_CREATE_CONTAINER;

    /******************************************************************************************************************
     * Show window
     *****************************************************************************************************************/
    HRESULT hr = S_OK;
    {
        WACore::RevertCast<CRCWindowAttr, WACore::IContainable> window(gContainer->Get(idMainWindowAttr));
        if (!window()) return CRC::ERROR_CAST;

        hr = CRC::ShowWindowCRC(window()->hWnd_);
        if (FAILED(hr)) return CRC::ERROR_SHOW_WINDOW;
    }

    /******************************************************************************************************************
     * Create window message event caller.
     *****************************************************************************************************************/
    HWND key;
    {
        WACore::RevertCast<CRCWindowAttr, WACore::IContainable> window(gContainer->Get(idMainWindowAttr));
        if (!window()) return CRC::ERROR_CAST;

        key = window()->hWnd_;
    }

    WACore::AddWPEventInsts
    (
        key, 
        std::make_unique<MainWindowEvent>(idMainWindowAttr, idMainSceneAttr, idUserInputAttr), 
        gWPEventCaller.instTable_
    );
    WACore::AddWPEventInsts
    (
        key, 
        std::make_unique<MainSceneEvent>(idMainWindowAttr, idMainSceneAttr, idUserInputAttr), 
        gWPEventCaller.instTable_
    );
    WACore::AddWPEventInsts
    (
        key, 
        std::make_unique<CRCUserInputEvent>(idUserInputAttr), 
        gWPEventCaller.instTable_
    );

    WACore::AddWPEventFuncs(gWPEventCaller.funcTable_);
    
    /******************************************************************************************************************
     * Main loop.
     *****************************************************************************************************************/
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
}

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    gWPEventCaller.Call(hWnd, msg, gContainer, msg, wParam, lParam);

    switch (msg)
    {
    case WM_PAINT:
        // Returning DefWindowProc with WM_PAINT will stop the updating, so WM_PAINT is returned 0.
        break;

    default:
        return DefWindowProc(hWnd, msg, wParam, lParam);
    }

    return 0;
}