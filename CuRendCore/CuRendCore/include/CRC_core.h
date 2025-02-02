#pragma once

#include <memory>
#include <vector>
#include <Windows.h>

#include "CRC_config.h"
#include "CRC_interface.h"

class CRCWindowContainer;
class CRCSceneContainer;

class CRC_API CRCCore
{
private:
    std::vector<std::unique_ptr<ICRCContainer>> containers_;

public:
    CRCCore();
    virtual ~CRCCore();

    // Delete copy constructor and operator=.
    CRCCore(const CRCCore&) = delete;
    CRCCore& operator=(const CRCCore&) = delete;

    virtual void Initialize();
    virtual int Shutdown();

    virtual HRESULT SetWindowContainer(std::unique_ptr<ICRCContainer> container);
    virtual HRESULT SetSceneContainer(std::unique_ptr<ICRCContainer> container);

    virtual HRESULT CreateWindowCRC(int idWindow);
    virtual HRESULT ShowWindowCRC(int idWindow);

    virtual HRESULT CreateScene(int idScene);

    virtual HRESULT SetSceneToWindow(int idWindow, int idScene);
};