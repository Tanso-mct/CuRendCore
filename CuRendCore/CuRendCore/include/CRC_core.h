#pragma once

#include "CRC_config.h"
#include "CRC_event_caller.h"

#include <memory>
#include <vector>
#include <Windows.h>
#include <unordered_map>

class CRCContainerSet;
class ICRCWinMsgListener;

class CRC_API CRCCore
{
public:
    CRCCore();
    virtual ~CRCCore();

    // Delete copy constructor and operator=.
    CRCCore(const CRCCore&) = delete;
    CRCCore& operator=(const CRCCore&) = delete;

    std::unique_ptr<CRCContainerSet> containerSet_;
    std::unique_ptr<CRCEventCaller<HWND, ICRCWinMsgListener, UINT, WPARAM, LPARAM>> winMsgCaller_;

    virtual void Initialize();
    virtual int Shutdown();

    virtual void HandleWindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual void FrameUpdate(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
};