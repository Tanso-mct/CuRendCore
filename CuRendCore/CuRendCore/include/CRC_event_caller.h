#pragma once

#include "CRC_config.h"
#include "CRC_container.h"

#include <unordered_map>

template <typename KEY, typename EVENT, typename... Args>
class CRC_API CRCEventCaller
{
private:
    std::unordered_map<KEY, std::vector<std::unique_ptr<EVENT>>> listeners_;

public:
    CRCEventCaller() = default;
    virtual ~CRCEventCaller() = default;

    // Delete copy constructor and operator=.
    CRCEventCaller(const CRCEventCaller&) = delete;
    CRCEventCaller& operator=(const CRCEventCaller&) = delete;

    void Add(KEY key, std::unique_ptr<EVENT> listener)
    {
        listeners_[key].emplace_back(std::move(listener));
    }

    HRESULT Clear(KEY key)
    {
        if (listeners_.find(key) == listeners_.end()) return E_FAIL;

        listeners_[key].clear();
        listeners_.erase(key);

        return S_OK;
    }

    void Call(KEY key, void (EVENT::*func)(Args...), Args... args)
    {
        for (std::size_t i = 0; i < listeners_[key].size(); ++i)
        {
            (listeners_[key][i].get()->*func)(args...);
        }
    }
};