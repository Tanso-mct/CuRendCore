#pragma once

#include "CuRendCore/include/config.h"
#include "CuRendCore/include/container.h"

#include <Windows.h>
#include <unordered_map>

template <typename KEY, typename EVENT_FUNC, typename CALLER>
class CRC_API CRCEventSet
{
public:
    std::unordered_map<KEY, EVENT_FUNC> funcMap_;
    std::unique_ptr<CALLER> caller_ = nullptr;

    CRCEventSet() = default;
    virtual ~CRCEventSet() = default;

    // Delete copy constructor and operator=.
    CRCEventSet(const CRCEventSet&) = delete;
    CRCEventSet& operator=(const CRCEventSet&) = delete;
};

class ICRCWinMsgEvent
{
public:
    virtual ~ICRCWinMsgEvent() = default;

    virtual void OnSetFocus(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnKillFocus(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnSize(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnUpdate(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnMove(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnDestroy(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnKeyDown(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnKeyUp(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam){};
    virtual void OnMouse(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam){};
};

namespace CRC
{

using WinMsgEventKey = UINT;
using WinMsgEventFunc = void (ICRCWinMsgEvent::*)(std::unique_ptr<ICRCContainer>&, UINT, WPARAM, LPARAM);

void CRC_API CreateWinMsgEventFuncMap(std::unordered_map<WinMsgEventKey, WinMsgEventFunc>& map);

}