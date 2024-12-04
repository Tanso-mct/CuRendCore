#pragma once

#include <memory>

#include "CRCConfig.h"

#include "Window.h"
#include "Scene.h"
#include "Resource.cuh"
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
    SceneFactory* sceneFc;
    ResourceFactory* resourceFc;
    BinderFactory* binderFc;

    int Run(HINSTANCE hInstance, int nCmdShow);
};

} // namespace CRC