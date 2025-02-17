#pragma once

#include "CRC_config.h"
#include "CRC_event_caller.h"

#include <memory>
#include <vector>
#include <Windows.h>
#include <unordered_map>

class CRCContainerSet;
class ICRCWinMsgEvent;

class CRC_API CRCCore
{
private:
    std::unordered_map<UINT, void (ICRCWinMsgEvent::*)(UINT, WPARAM, LPARAM)> handledMsgMap_;
    
public:
    CRCCore();
    virtual ~CRCCore();

    // Delete copy constructor and operator=.
    CRCCore(const CRCCore&) = delete;
    CRCCore& operator=(const CRCCore&) = delete;

    std::unique_ptr<CRCContainerSet> containerSet_;
    std::unique_ptr<CRCEventCaller<HWND, ICRCWinMsgEvent, UINT, WPARAM, LPARAM>> winMsgCaller_;

    virtual void Initialize();
    virtual int Shutdown();

    virtual void HandleWindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    virtual void FrameUpdate(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
};