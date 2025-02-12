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

    virtual void Update(ICRCContainable* attr) = 0;
    virtual void Hide(ICRCContainable* attr) = 0;
    virtual void Restored(ICRCContainable* attr) = 0;
    virtual void End(ICRCContainable* attr) = 0;
};

template <typename KEY>
class CRC_API CRCPhaseMethodCaller
{
private:
    std::unordered_map<KEY, std::vector<std::unique_ptr<ICRCPhaseMethod>>> pms_;
    std::unordered_map<KEY, std::vector<ICRCContainable*>> attrs_;

public:
    CRCPhaseMethodCaller() = default;
    virtual ~CRCPhaseMethodCaller() = default;

    // Delete copy constructor and operator=.
    CRCPhaseMethodCaller(const CRCPhaseMethodCaller&) = delete;
    CRCPhaseMethodCaller& operator=(const CRCPhaseMethodCaller&) = delete;

    void Add(std::unique_ptr<ICRCPhaseMethod> phaseMethod, ICRCContainable* attr, KEY key)
    {
        pms_[key].emplace_back(std::move(phaseMethod));
        attrs_[key].emplace_back(attr);
    }

    HRESULT Clear(KEY key)
    {
        if (pms_.find(key) == pms_.end()) return E_FAIL;

        pms_[key].clear();
        attrs_[key].clear();

        pms_.erase(key);
        attrs_.erase(key);

        return S_OK;
    }

    void CallUpdate(KEY key){ for (int i = 0; i < pms_[key].size(); ++i){ pms_[key][i]->Update(attrs_[key][i]); } }
    void CallHide(KEY key){ for (int i = 0; i < pms_[key].size(); ++i){ pms_[key][i]->Hide(attrs_[key][i]); } }
    void CallRestored(KEY key){ for (int i = 0; i < pms_[key].size(); ++i){ pms_[key][i]->Restored(attrs_[key][i]); } }
    void CallEnd(KEY key){ for (int i = 0; i < pms_[key].size(); ++i){ pms_[key][i]->End(attrs_[key][i]); } }
};