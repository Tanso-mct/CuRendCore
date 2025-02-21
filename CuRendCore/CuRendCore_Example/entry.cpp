#include "CRC_pch.h"

#include "CuRendCore.h"

#include <Windows.h>

#include "window.h"
#include "scene.h"

static CRC::WinMsgEventSet WinMsgEventSet;
static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

int main() 
{
    /******************************************************************************************************************
     * Create attributes container.
     *****************************************************************************************************************/
    std::unique_ptr<ICRCContainer> container = std::make_unique<CRCContainer>();

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
        std::unique_ptr<ICRCContainable> windowAttr = windowFactory.Create(desc);

        // Add window attribute to container.
        idMainWindowAttr = container->Add(std::move(windowAttr));
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
        std::unique_ptr<ICRCContainable> sceneAttr = sceneFactory.Create(desc);

        // Add scene to scene container.
        idMainSceneAttr = container->Add(std::move(sceneAttr));
    }
    if (idMainSceneAttr == CRC::ID_INVALID) return CRC::ERROR_ADD_TO_CONTAINER;

    /******************************************************************************************************************
     * User Input attributes creation.
     *****************************************************************************************************************/
    int idUserInputAttr = CRC::ID_INVALID;
    {
        // Create user input by user input attributes.
        std::unique_ptr<ICRCContainable> userInputAttr = std::make_unique<CRCUserInputAttr>();

        // Add user input to user input container.
        idUserInputAttr = container->Add(std::move(userInputAttr));
    }
    if (idUserInputAttr == CRC::ID_INVALID) return CRC::ERROR_CREATE_CONTAINER;

    /******************************************************************************************************************
     * Show window
     *****************************************************************************************************************/
    HRESULT hr = S_OK;

    hr = CRC::ShowWindowCRC(CRC::As<CRCWindowAttr>(container->Get(idMainWindowAttr).get())->hWnd_);
    if (FAILED(hr)) return CRC::ERROR_SHOW_WINDOW;

    /******************************************************************************************************************
     * Create window message event set.
     *****************************************************************************************************************/
    CRC::CreateWinMsgEventFuncMap(WinMsgEventSet.funcMap_);
    WinMsgEventSet.caller_ = std::make_unique<CRC::WinMsgEventCaller>();

    /******************************************************************************************************************
     * Create a Window Message event and
     * register it with the window message event caller.
     *****************************************************************************************************************/
    {
        // Set key to windows message event caller.
        HWND key = CRC::As<CRCWindowAttr>(container->Get(idMainWindowAttr).get())->hWnd_;
        WinMsgEventSet.caller_->AddKey(key);

        // Move container to windows message event caller.
        WinMsgEventSet.caller_->MoveContainer(key, std::move(container));

        /**
         * Add event to windows message event caller.
         * when processing the window's message.
         * Also, the order in which event are called is the order in which they are added to winMsgCaller_.
         */
        WinMsgEventSet.caller_->AddEvent(key, std::make_unique<MainWindowEvent>
        (
            idMainWindowAttr, idMainSceneAttr, idUserInputAttr
        ));

        WinMsgEventSet.caller_->AddEvent(key, std::make_unique<MainSceneEvent>
        (
            idMainWindowAttr, idMainSceneAttr, idUserInputAttr
        ));

        WinMsgEventSet.caller_->AddEvent(key, std::make_unique<CRCUserInputEvent>
        (
            idUserInputAttr
        ));
    }
    
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
    CRC::CallWinMsgEvent(WinMsgEventSet, hWnd, msg, wParam, lParam);

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