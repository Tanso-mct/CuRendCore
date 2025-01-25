#include "CRC_pch.h"

#include "CuRendCore.h"

int APIENTRY WinMain
(
    _In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPSTR lpCmdLine, _In_ int nCmdShow
) {
    std::unique_ptr<CRCCore> core = CRC::CreateCRCCore();
    core->Initialize();

    int idMainWindow = -1;
    {
        std::unique_ptr<CRCContainer> windowContainer = CRC::CreateWindowContainer();

        CRCWindowAttr windowAttr;
        windowAttr.width = 800;
        windowAttr.height = 600;
        std::unique_ptr<CRCData> windowData = CRC::CreateWindowData(windowAttr);
        idMainWindow = windowContainer->Add(windowData);
        
        core->SetWindowContainer(windowContainer);
    }

    core->Run();

    return core->Shutdown();
}