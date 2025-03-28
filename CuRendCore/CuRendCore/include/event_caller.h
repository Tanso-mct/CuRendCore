#pragma once

#include "CuRendCore/include/config.h"
#include "CuRendCore/include/container.h"
#include "CuRendCore/include/event.h"

#include <unordered_map>

template <typename KEY, typename EVENT, typename... Args>
class CRC_API CRCEventCaller
{
private:
    std::unordered_map
    <
        KEY, 
        std::pair<std::vector<std::unique_ptr<EVENT>>, std::unique_ptr<ICRCContainer>>
    > events_;

public:
    CRCEventCaller() = default;
    virtual ~CRCEventCaller() = default;

    // Delete copy constructor and operator=.
    CRCEventCaller(const CRCEventCaller&) = delete;
    CRCEventCaller& operator=(const CRCEventCaller&) = delete;

    HRESULT AddKey(KEY key)
    {
        if (events_.find(key) != events_.end()) return E_FAIL;

        events_[key] = std::make_pair(std::vector<std::unique_ptr<EVENT>>(), nullptr);
        return S_OK;
    }

    int AddEvent(KEY key, std::unique_ptr<EVENT> listener)
    {   
        if (events_.find(key) == events_.end()) return CRC::ID_INVALID;

        events_[key].first.emplace_back(std::move(listener));
        return events_[key].first.size() - 1;
    }

    HRESULT MoveContainer(KEY key, std::unique_ptr<ICRCContainer> container)
    {
        if (events_.find(key) == events_.end()) return E_FAIL;

        events_[key].second = std::move(container);
        return S_OK;
    }

    std::unique_ptr<EVENT> TakeEvent(KEY key, int id)
    {
        if (events_.find(key) == events_.end()) return nullptr;
        if (id < 0 || id >= events_[key].first.size()) return nullptr;

        std::unique_ptr<EVENT> event = std::move(events_[key].first[id]);
        return event;
    }

    std::unique_ptr<ICRCContainer> TakeContainer(KEY key)
    {
        if (events_.find(key) == events_.end()) return nullptr;

        std::unique_ptr<ICRCContainer> container = std::move(events_[key].second);
        return container;
    }

    void Call(KEY key, void (EVENT::*func)(std::unique_ptr<ICRCContainer>&, Args...), Args... args)
    {
        for (int i = 0; i < events_[key].first.size(); i++)
        {
            if (events_[key].first[i] == nullptr) continue;
            (events_[key].first[i].get()->*func)(events_[key].second, args...);
        }
    }
};

namespace CRC
{

using WinMsgEventCaller = CRCEventCaller<HWND, ICRCWinMsgEvent, UINT, WPARAM, LPARAM>;
using WinMsgEventSet = CRCEventSet<WinMsgEventKey, WinMsgEventFunc, WinMsgEventCaller>;

CRC_API void CallWinMsgEvent
(
    WinMsgEventSet& eventSet, 
    HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam
);

}