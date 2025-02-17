#include "CRC_pch.h"

#include "CuRendCore.h"

#include <Windows.h>

#include "window.h"
#include "scene.h"
#include "slot_id.h"

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

int main() 
{
    /**************************************************************************************************************** */
    // Core initialization.
    /**************************************************************************************************************** */
    CRC::Core()->Initialize();

    /**************************************************************************************************************** */
    // Window attributes creation.
    /**************************************************************************************************************** */
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
        SlotID::MAIN_WINDOW = container->Add(std::move(windowData));

        // Move window container to core.
        SlotID::MAIN_WINDOW_CONTAINER = CRC::Core()->containerSet_->Add(std::move(container));
    }
    if (SlotID::MAIN_WINDOW == CRC::ID_INVALID) return CRC::ERROR_CREATE_WINDOW;
    if (SlotID::MAIN_WINDOW_CONTAINER == CRC::ID_INVALID) return CRC::ERROR_CREATE_CONTAINER;


    /**************************************************************************************************************** */
    // Scene attributes creation.
    /**************************************************************************************************************** */
    {
        // Create scene container.
        std::unique_ptr<ICRCContainer> container = std::make_unique<CRCContainer>();

        // Create scene by scene attributes.
        std::unique_ptr<CRCSceneSrc> sceneAttr = std::make_unique<CRCSceneSrc>();
        sceneAttr->name_ = "MainScene";
        std::unique_ptr<ICRCContainable> sceneData = CRC::CreateSceneAttr(std::move(sceneAttr));

        // Add scene to scene container.
        SlotID::MAIN_SCENE = container->Add(std::move(sceneData));

        // Move scene container to core.
        SlotID::MAIN_SCENE_CONTAINER = CRC::Core()->containerSet_->Add(std::move(container));
    }
    if (SlotID::MAIN_SCENE == CRC::ID_INVALID) return CRC::ERROR_CREATE_CONTAINER;
    if (SlotID::MAIN_SCENE_CONTAINER == CRC::ID_INVALID) return CRC::ERROR_CREATE_CONTAINER;

    /**************************************************************************************************************** */
    // User Input attributes creation.
    /**************************************************************************************************************** */
    {
        // Create user input container.
        std::unique_ptr<ICRCContainer> container = std::make_unique<CRCContainer>();

        // Create user input by user input attributes.
        std::unique_ptr<ICRCContainable> userInputAttr = std::make_unique<CRCUserInputAttr>();

        // Add user input to user input container.
        SlotID::USER_INPUT = container->Add(std::move(userInputAttr));

        // Move user input container to core.
        SlotID::USER_INPUT_CONTAINER = CRC::Core()->containerSet_->Add(std::move(container));
    }
    if (SlotID::USER_INPUT == CRC::ID_INVALID) return CRC::ERROR_CREATE_CONTAINER;
    if (SlotID::USER_INPUT_CONTAINER == CRC::ID_INVALID) return CRC::ERROR_CREATE_CONTAINER;

    /**************************************************************************************************************** */
    // Create window, show window, create scene, set scene to window.
    /**************************************************************************************************************** */
    HRESULT hr = S_OK;

    // Create window.
    hr = CRC::CreateWindowCRC(CRC::GetContainable(SlotID::MAIN_WINDOW_CONTAINER, SlotID::MAIN_WINDOW));
    if (FAILED(hr)) return CRC::ERROR_CREATE_WINDOW;

    // Create scene.
    hr = CRC::CreateScene(CRC::GetContainable(SlotID::MAIN_SCENE_CONTAINER, SlotID::MAIN_SCENE));
    if (FAILED(hr)) return CRC::ERROR_CREATE_SCENE;

    /**************************************************************************************************************** */
    // Set Windows Message Event to window.
    /**************************************************************************************************************** */

    /**
     * when processing the window's message.
     * Also, the order in which event are called is the order in which they are added to winMsgCaller_.
     */

    {
        CRCWindowAttr* mainWindowAttr = CRC::GetContainablePtr<CRCWindowAttr>
        (
            SlotID::MAIN_WINDOW_CONTAINER, SlotID::MAIN_WINDOW
        );

        CRC::Core()->winMsgCaller_->Add // Window Event.
        (
            mainWindowAttr->hWnd_, // Key.
            std::make_unique<MainWindowListener>()
        );

        CRC::Core()->winMsgCaller_->Add // Scene Event.
        (
            mainWindowAttr->hWnd_, // Key.
            std::make_unique<MainSceneListener>()
        );

        CRC::Core()->winMsgCaller_->Add // User Input Event.
        (
            mainWindowAttr->hWnd_, // Key.
            std::make_unique<CRCUserInputListener>(SlotID::USER_INPUT_CONTAINER, SlotID::USER_INPUT)
        );
    }

    /**************************************************************************************************************** */
    // Main loop.
    /**************************************************************************************************************** */

    // Show window.
    hr = CRC::ShowWindowCRC(CRC::GetContainable(SlotID::MAIN_WINDOW_CONTAINER, SlotID::MAIN_WINDOW));
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