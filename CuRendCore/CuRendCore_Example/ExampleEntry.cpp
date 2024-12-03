#include "CuRendCore.h"

#include "ExampleWndCtrl.h"
#include "ExampleSceneCtrl.h"

int WINAPI WinMain
(
    _In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPSTR lpCmdLine, _In_ int nCmdShow
) {
    // Get the instance of the CuRendCore
    CRC::CuRendCore* crc = CRC::CuRendCore::GetInstance();

    {
        // Create Window
        CRC::WNDATTR wattr;
        wattr.wcex.cbSize = sizeof(WNDCLASSEX);
        wattr.wcex.style = CS_HREDRAW | CS_VREDRAW;
        wattr.wcex.cbClsExtra = NULL;
        wattr.wcex.cbWndExtra = NULL;
        wattr.wcex.hInstance = hInstance;
        wattr.wcex.hIcon = LoadIcon(NULL, IDI_APPLICATION);
        wattr.wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
        wattr.wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        wattr.wcex.lpszMenuName = NULL;
        wattr.wcex.lpszClassName = L"WindowClass";
        wattr.wcex.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
        wattr.hInstance = hInstance;
        wattr.ctrl = std::make_shared<ExampleWndCtrl>();

        CRC_SLOT slotExampleWnd = crc->windowFc->CreateWindowCRC(wattr);
        crc->windowFc->ShowWindowCRC(0, nCmdShow);

        // Create Scene
        CRC::SCENEATTR sattr;

        std::shared_ptr<ExampleSceneCtrl> exampleSceneCtrl = std::make_shared<ExampleSceneCtrl>();
        sattr.name = "ExampleScene";
        sattr.ctrl = exampleSceneCtrl;
        crc->sceneFc->CreateScene(sattr);

        std::shared_ptr<ExampleScene2Ctrl> exampleScene2Ctrl = std::make_shared<ExampleScene2Ctrl>();
        sattr.name = "ExampleScene2";
        sattr.ctrl = exampleScene2Ctrl;
        crc->sceneFc->CreateScene(sattr);

        exampleSceneCtrl->SetSlotWnd(slotExampleWnd);

        // Avoid duplicating shared_ptr and use weak_ptr since the data is managed by CuRendCore.
        exampleSceneCtrl->SetExampleScene2Ctrl(exampleScene2Ctrl);

        // Set the Scene controller to the Window
        crc->windowFc->SetSceneCtrl(slotExampleWnd, exampleSceneCtrl);
    }

    // Run the CuRendCore
    return crc->Run(hInstance, nCmdShow);
}