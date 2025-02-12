#pragma once

#include "CRC_config.h"
#include "CRC_phase_method.h"

#include <memory>
#include <vector>
#include <Windows.h>
#include <unordered_map>

class CRCContainerSet;

class CRC_API CRCCore
{
public:
    CRCCore();
    virtual ~CRCCore();

    // Delete copy constructor and operator=.
    CRCCore(const CRCCore&) = delete;
    CRCCore& operator=(const CRCCore&) = delete;

    std::unique_ptr<CRCContainerSet> containerSet_;
    std::unique_ptr<CRCPhaseMethodCaller<HWND>> pmCaller_;

    virtual void Initialize();
    virtual int Shutdown();
    virtual void HandleWindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
};