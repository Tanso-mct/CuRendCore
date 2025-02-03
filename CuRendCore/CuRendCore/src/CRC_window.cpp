#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_window.h"

int CRCWindowContainer::Add(std::unique_ptr<ICRCContainable> data)
{
    std::unique_ptr<CRCWindowAttr> windowData = CRC::UniqueAs<CRCWindowAttr>(data);

    if (windowData)
    {
        data_.push_back(std::move(windowData));
        return data_.size() - 1;
    }
    else return CRC::ID_INVALID;
}

HRESULT CRCWindowContainer::Remove(int id)
{
    if (id < 0 || id >= data_.size()) return E_FAIL;

    data_[id].reset();
    return S_OK;
}

ICRCContainable* CRCWindowContainer::Get(int id)
{
    if (id < 0 || id >= data_.size()) return nullptr;

    return CRC::PtrAs<ICRCContainable>(data_[id].get());
}

int CRCWindowContainer::GetSize()
{
    return data_.size();
}

void CRCWindowContainer::Clear()
{
    data_.clear();
}