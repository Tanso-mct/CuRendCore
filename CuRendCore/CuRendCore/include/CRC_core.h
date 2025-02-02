#pragma once

#include <memory>
#include <Windows.h>

#include "CRC_config.h"
#include "CRC_interface.h"

class CRCWindowContainer;
class CRCSceneContainer;

class CRC_API CRCCore
{
private:
    std::unique_ptr<CRCWindowContainer> windowContainer_ = nullptr;
    std::unique_ptr<CRCSceneContainer> sceneContainer_ = nullptr;

public:
    CRCCore();
    virtual ~CRCCore();

    virtual void Initialize();
    virtual int Shutdown();

    virtual HRESULT SetWindowContainer(std::unique_ptr<CRCContainer> container);
    virtual HRESULT SetSceneContainer(std::unique_ptr<CRCContainer> container);

    virtual HRESULT SetSceneToWindow(int idWindow, int idScene);
};