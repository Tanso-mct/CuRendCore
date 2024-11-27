#pragma once

#include <memory>

#include "CRCConfig.h"

#include "Window.h"
#include "Scene.h"
#include "Resource.h"
#include "Binder.h"

namespace CRC 
{

class CRC_API CuRendCore 
{
private:
    CuRendCore(); // Singleton pattern.

public:
    ~CuRendCore();

    static CuRendCore* GetInstance();
    void ReleaseInstance();

    // Various factories. Managed collectively in this class.
    WindowFactory* windowFc;
    std::shared_ptr<SceneFactory> sceneFc;
    std::shared_ptr<ResourceFactory> resourceFc;
    std::shared_ptr<BinderFactory> binderFc;

    int Run(HINSTANCE hInstance, int nCmdShow);
};

} // namespace CRC