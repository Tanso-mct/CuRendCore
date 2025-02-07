#pragma once

#include <memory>
#include <vector>
#include <Windows.h>
#include <unordered_map>
#include <tuple>

#include "CRC_config.h"
#include "CRC_interface.h"
#include "CRC_container.h"

class CRC_API CRCCore
{
private:
    std::unordered_map<HWND, std::pair<std::unique_ptr<ICRCPhaseMethod>, std::unique_ptr<ICRCPhaseMethod>>> hWndPMs;

public:
    CRCCore();
    virtual ~CRCCore();

    // Delete copy constructor and operator=.
    CRCCore(const CRCCore&) = delete;
    CRCCore& operator=(const CRCCore&) = delete;

    virtual void Initialize();
    virtual int Shutdown();

    virtual void HandleWindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
};