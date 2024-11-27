#include "CuRendCore.h"

namespace CRC 
{

CuRendCore::CuRendCore() 
{
    // Initialize the factories.
    windowFc = WindowFactory::GetInstance();
    sceneFc = std::make_shared<SceneFactory>();
    resourceFc = std::make_shared<ResourceFactory>();
    binderFc = std::make_shared<BinderFactory>();
}

CuRendCore::~CuRendCore()
{
    // Release the factories.
    if (windowFc != nullptr) delete windowFc;
    if (sceneFc != nullptr) sceneFc.reset();
    if (resourceFc != nullptr) resourceFc.reset();
    if (binderFc != nullptr) binderFc.reset();
}

CuRendCore* CuRendCore::GetInstance()
{
    // Implementation of the Singleton pattern.
    static CuRendCore* instance = nullptr;

    if (instance == nullptr)
    {
        instance = new CuRendCore();
    }

    return instance;
}

void CuRendCore::ReleaseInstance()
{
    // Implementation of the Singleton pattern.
    CuRendCore* instance = GetInstance();
    if (instance != nullptr) delete instance;
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