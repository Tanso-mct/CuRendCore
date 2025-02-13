#pragma once

#include "CRC_config.h"
#include "CRC_container.h"

#include <unordered_map>

template <typename KEY, typename LISTENER, typename... Args>
class CRC_API CRCEventCaller
{
private:
    std::unordered_map<KEY, std::vector<std::unique_ptr<LISTENER>>> listeners_;
    std::unordered_map<KEY, std::vector<ICRCContainable*>> attrs_;

public:
    CRCEventCaller() = default;
    virtual ~CRCEventCaller() = default;

    // Delete copy constructor and operator=.
    CRCEventCaller(const CRCEventCaller&) = delete;
    CRCEventCaller& operator=(const CRCEventCaller&) = delete;

    void Add(KEY key, std::unique_ptr<LISTENER> listener, ICRCContainable* attr)
    {
        listeners_[key].emplace_back(std::move(listener));
        attrs_[key].emplace_back(attr);
    }

    HRESULT Clear(KEY key)
    {
        if (listeners_.find(key) == listeners_.end()) return E_FAIL;

        listeners_[key].clear();
        attrs_[key].clear();

        listeners_.erase(key);
        attrs_.erase(key);

        return S_OK;
    }

    void Call(KEY key, void (LISTENER::*func)(ICRCContainable*, Args...), Args... args)
    {
        for (std::size_t i = 0; i < listeners_[key].size(); ++i)
        {
            (listeners_[key][i].get()->*func)(attrs_[key][i], args...);
        }
    }
};