#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_event_caller.h"

CRC_API void CRC::CallWinMsgEvent
(
    WinMsgEventSet &eventSet, 
    HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam
){
    if (eventSet.funcMap_.find(msg) == eventSet.funcMap_.end()) return;
    eventSet.caller_->Call(hWnd, eventSet.funcMap_[msg], msg, wParam, lParam);
}
