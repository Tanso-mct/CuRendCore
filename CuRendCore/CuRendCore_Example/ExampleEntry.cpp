#include "CuRendCore.h"

#include "ExampleWndCtrl.h"

int WINAPI WinMain
(
    _In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPSTR lpCmdLine, _In_ int nCmdShow
) {
    // Get the instance of the CuRendCore
    CRC::CuRendCore* crc = CRC::CuRendCore::GetInstance();

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
    wattr.ctrl = std::make_unique<ExampleWndCtrl>(new ExampleWndCtrl());

    CRC_SLOT slotExampleWnd = crc->windowFc->CreateWindowCRC(wattr);
    crc->windowFc->ShowWindowCRC(0, nCmdShow);

    // Create Scene
    CRC::SCENEATTR sattr;

    sattr.name = "ExampleScene";
    CRC_SLOT slotExampleScene1 = crc->sceneFc->CreateScene(sattr);

    sattr.name = "ExampleScene2";
    CRC_SLOT slotExampleScene2 = crc->sceneFc->CreateScene(sattr);

    // Set the scene
    crc->windowFc->SetScene(slotExampleWnd, slotExampleScene1);

    // Run the CuRendCore
    return crc->Run(hInstance, nCmdShow);
}