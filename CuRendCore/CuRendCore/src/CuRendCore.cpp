#include "CuRendCore.h"

namespace CRC 
{

CuRendCore::CuRendCore() 
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    // Initialize the factories.
    windowFc = new WindowFactory();
    sceneFc = new SceneFactory();
    resourceFc = new ResourceFactory();
    // binderFc = BinderFactory::GetInstance();
}

CuRendCore::~CuRendCore()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    // Release the factories.
    if (windowFc != nullptr) delete windowFc;
    if (sceneFc != nullptr) delete sceneFc;
    if (resourceFc != nullptr) delete resourceFc;
    // if (binderFc != nullptr) binderFc->ReleaseInstance();
}

CuRendCore* CuRendCore::GetInstance()
{
    // Implementation of the Singleton pattern.
    static CuRendCore* instance = nullptr;

    if (instance == nullptr) instance = new CuRendCore();

    return instance;
}

void CuRendCore::ReleaseInstance()
{
    // Implementation of the Singleton pattern.
    CuRendCore* instance = GetInstance();
    if (instance != nullptr)
    {
        delete instance;
        instance = nullptr;
    }
}

int CuRendCore::Run(HINSTANCE hInstance, int nCmdShow)
{
    // Main loop.
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

    ReleaseInstance();

    // Return this part of the WM_QUIT message to Windows.
    return static_cast<char>(msg.wParam);
}

} // namespace CRC