#pragma once

#include <memory>
#include <vector>
#include <Windows.h>
#include <unordered_map>


#include "CRC_config.h"
#include "CRC_interface.h"
#include "CRC_container.h"

class CRC_API CRCCore
{
private:
    std::vector<std::unique_ptr<ICRCContainer>> containers_;
    std::unique_ptr<ICRCContainer> emptyContainer_ = nullptr;

    std::unordered_map<HWND, std::vector<std::unique_ptr<ICRCPhaseMethod>>> hWndPMs;
    std::unique_ptr<ICRCPhaseMethod> emptyPhaseMethod_ = nullptr;

public:
    CRCCore();
    virtual ~CRCCore();

    // Delete copy constructor and operator=.
    CRCCore(const CRCCore&) = delete;
    CRCCore& operator=(const CRCCore&) = delete;

    virtual void Initialize();
    virtual int Shutdown();

    virtual int AddContainer(std::unique_ptr<ICRCContainer> container);
    virtual std::unique_ptr<ICRCContainer>& GetContainer(int id);

    virtual std::unique_ptr<ICRCContainer> TakeContainer(int id);
    virtual HRESULT PutContainer(int id, std::unique_ptr<ICRCContainer> container);

    virtual HRESULT SetSceneToWindow(std::unique_ptr<ICRCContainable>& windowAttr,int idScene, int idSceneContainer);
    virtual HRESULT AddPhaseMethodToWindow(HWND hWnd, std::unique_ptr<ICRCPhaseMethod> phaseMethod);

    virtual void HandleWindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
};