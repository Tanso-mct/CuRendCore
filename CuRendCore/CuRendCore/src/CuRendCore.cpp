#include "CuRendCore.h"

namespace CRC 
{

CuRendCore::CuRendCore() 
{
    // Initialize the factories.
    windowFc = WindowFactory::GetInstance();
    sceneFc = SceneFactory::GetInstance();
    // resourceFc = ResourceFactory::GetInstance();
    // binderFc = BinderFactory::GetInstance();

    // Initialize the input.
    input = Input::GetInstance();
}

CuRendCore::~CuRendCore()
{
    // Release the factories.
    if (windowFc != nullptr) windowFc->ReleaseInstance();
    if (sceneFc != nullptr) sceneFc->ReleaseInstance();
    // if (resourceFc != nullptr) resourceFc->ReleaseInstance();
    // if (binderFc != nullptr) binderFc->ReleaseInstance();

    // Release the input.
    if (input != nullptr) input->ReleaseInstance();
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