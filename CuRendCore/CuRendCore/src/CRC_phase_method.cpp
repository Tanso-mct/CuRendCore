#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_phase_method.h"

int CRCPMContainer::Add(std::unique_ptr<ICRCContainable> data)
{
    std::unique_ptr<ICRCPhaseMethod> phaseMethod = CRC::UniqueAs<ICRCPhaseMethod>(data);

    if (phaseMethod)
    {
        methods_.push_back(std::move(phaseMethod));
        return methods_.size() - 1;
    }
    else return CRC::ID_INVALID;
}

HRESULT CRCPMContainer::Remove(int id)
{
    if (id < 0 || id >= methods_.size()) return E_FAIL;
    
    methods_[id].reset();
    return S_OK;
}

ICRCContainable *CRCPMContainer::Get(int id)
{
    if (id < 0 || id >= methods_.size()) return nullptr;
    return CRC::PtrAs<ICRCContainable>(methods_[id].get());
}

int CRCPMContainer::GetSize()
{
    return methods_.size();
}

void CRCPMContainer::Clear()
{
    methods_.clear();
}
