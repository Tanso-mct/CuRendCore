#pragma once

#include <memory>
#include <vector>
#include <Windows.h>
#include <unordered_map>
#include <tuple>

#include "CRC_config.h"
#include "CRC_interface.h"

class CRC_API CRCCore
{
private:
    std::vector<std::unique_ptr<ICRCContainer>> containers_;
    std::unordered_map<HWND, std::tuple<ICRCContainable*, ICRCPhaseMethod*, ICRCPhaseMethod*>> existWindows_;

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
    virtual HRESULT SetWindowPMContainer(std::unique_ptr<ICRCContainer> container);
    virtual HRESULT SetScenePMContainer(std::unique_ptr<ICRCContainer> container);

    virtual HRESULT CreateWindowCRC(int idWindow, int idWindowPM);
    virtual HRESULT ShowWindowCRC(int idWindow);

    virtual HRESULT CreateScene(int idScene, int idScenePM);
    virtual HRESULT SetSceneToWindow(int idWindow, int idScene);

    virtual void HandleWindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
};