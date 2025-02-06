#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_container.h"

int CRCContainer::Add(std::unique_ptr<ICRCContainable> data)
{
    datas_.emplace_back(std::move(data));
    return datas_.size() - 1;
}

HRESULT CRCContainer::Remove(int id)
{
    if (id < 0 || id >= datas_.size()) return E_FAIL;

    datas_[id].reset();
    return S_OK;
}

ICRCContainable *CRCContainer::Get(int id)
{
    if (id < 0 || id >= datas_.size()) return nullptr;
    return datas_[id].get();
}

int CRCContainer::GetSize()
{
    return datas_.size();
}

void CRCContainer::Clear()
{
    datas_.clear();
}
