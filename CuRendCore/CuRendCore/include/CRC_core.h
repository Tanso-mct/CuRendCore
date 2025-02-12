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
    std::unordered_map<HWND, std::vector<std::unique_ptr<ICRCPhaseMethod>>> hWndPMs;
    std::unique_ptr<ICRCPhaseMethod> emptyPhaseMethod_ = nullptr;

public:
    CRCCore();
    virtual ~CRCCore();

    // Delete copy constructor and operator=.
    CRCCore(const CRCCore&) = delete;
    CRCCore& operator=(const CRCCore&) = delete;

    std::unique_ptr<CRCContainerSet> containers_;

    virtual void Initialize();
    virtual int Shutdown();

    virtual HRESULT AddPhaseMethodToWindow(HWND hWnd, std::unique_ptr<ICRCPhaseMethod> phaseMethod);

    virtual void HandleWindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
};