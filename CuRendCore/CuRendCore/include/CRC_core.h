#pragma once

#include <memory>
#include <Windows.h>

#include "CRC_config.h"
#include "CRC_interface.h"

#include "CRC_thread.h"

class CRCWindowContainer;
class CRCSceneContainer;

class CRCCore
{
private:
    std::unique_ptr<CRCWindowContainer> windowContainer_ = nullptr;
    std::unique_ptr<CRCSceneContainer> sceneContainer_ = nullptr;

    CRCThread threadWindow_;
    static void WindowThread(CRCThread& thread, std::unique_ptr<CRCWindowContainer> &container);

    CRCThread threadScene_;
    static void SceneThread(CRCThread& thread, std::unique_ptr<CRCSceneContainer> &container);

public:
    CRCCore() = default;
    virtual ~CRCCore() = default;

    virtual void Initialize();
    virtual int Shutdown();

    virtual std::unique_ptr<CRCContainer>& WindowContainer();
    virtual std::unique_ptr<CRCContainer>& SceneContainer();

    virtual void SendCmdToWindowThrd(int cmd);
    virtual int ReceiveCmdFromWindowThrd();
    virtual void GetResultFromWindowThrd(int& didCmd, int& error);

    virtual void SendCmdToSceneThrd(int cmd);
    virtual int ReceiveCmdFromSceneThrd();
    virtual void GetResultFromSceneThrd(int& didCmd, int& error);

    virtual HRESULT SetSceneToWindow(int idWindow, int idScene);
};