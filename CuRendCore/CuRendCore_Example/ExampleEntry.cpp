#include "CuRendCore.h"

#include "Slots.h"
#include "ExampleWindow.h"
#include "ExampleScene.h"

int WINAPI WinMain
(
    _In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPSTR lpCmdLine, _In_ int nCmdShow
) {
    // Get the instance of the CuRendCore
    CRC::CuRendCore* crc = CRC::CuRendCore::GetInstance();

    // Create Window
    CRC::WND_ATTR wattr;
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

    ExampleWndCtrl* ewc = new ExampleWndCtrl();
    wattr.ctrl = std::unique_ptr<ExampleWndCtrl>(ewc);

    Slots::EXAMPLE_WINDOW = crc->windowFc->CreateWindowCRC(wattr);
    crc->windowFc->ShowWindowCRC(0, nCmdShow);

    // Create Scene
    CRC::SCENE_ATTR sattr;

    sattr.name = "ExampleScene";
    sattr.sceneMani = new ExampleSceneMani();
    Slots::EXAMPLE_SCENE = crc->sceneFc->CreateScene(sattr);

    // Set the scene
    crc->windowFc->SetScene(Slots::EXAMPLE_WINDOW, Slots::EXAMPLE_SCENE);

    // Run the CuRendCore
    return crc->Run(hInstance, nCmdShow);
}