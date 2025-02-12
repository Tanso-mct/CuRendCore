#pragma once

#include "CRC_config.h"
#include "CRC_container.h"

#include <memory>
#include <vector>
#include <unordered_map>
#include <Windows.h>

class CRC_API ICRCPhaseMethod : public ICRCContainable
{
public:
    virtual ~ICRCPhaseMethod() = default;

    virtual void Update() = 0;
    virtual void Hide() = 0;
    virtual void Restored() = 0;
    virtual void End() = 0;
};

template <typename KEY>
class CRC_API CRCPhaseMethodCaller
{
private:
    std::unordered_map<KEY, std::vector<std::unique_ptr<ICRCPhaseMethod>>> pms;
    std::unordered_map<KEY, ICRCContainable*> attrs;

public:
    CRCPhaseMethodCaller() = default;
    virtual ~CRCPhaseMethodCaller() = default;

    // Delete copy constructor and operator=.
    CRCPhaseMethodCaller(const CRCPhaseMethodCaller&) = delete;
    CRCPhaseMethodCaller& operator=(const CRCPhaseMethodCaller&) = delete;

    void AddPhaseMethod(std::unique_ptr<ICRCPhaseMethod> phaseMethod, ICRCContainable* attr, KEY key)
    {
        pms[key].emplace_back(std::move(phaseMethod));
        attrs[key] = attr;
    }

    HRESULT ClearPhaseMethod(KEY key)
    {
        if (pms.find(key) == pms.end()) return E_FAIL;

        pms[key].clear();

        pms.erase(key);
        attrs.erase(key);

        return S_OK;
    }

    void CallUpdate(KEY key) { for (auto& pm : pms[key]) pm->Update(); }
    void CallHide(KEY key) { for (auto& pm : pms[key]) pm->Hide(); }
    void CallRestored(KEY key) { for (auto& pm : pms[key]) pm->Restored(); }
    void CallEnd(KEY key) { for (auto& pm : pms[key]) pm->End(); }
};