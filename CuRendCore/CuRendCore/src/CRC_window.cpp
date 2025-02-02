#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_window.h"

int CRCWindowContainer::Add(std::unique_ptr<CRCData>& data)
{
    std::unique_ptr<CRCWindowData> windowData = CRC::CastMove<CRCWindowData>(data);

    if (windowData)
    {
        data_.push_back(std::move(windowData));
        return data_.size() - 1;
    }
    else return CRC::INVALID_ID;
}

HRESULT CRCWindowContainer::Remove(int id)
{
    if (id < 0 || id >= data_.size()) return E_FAIL;

    data_.erase(data_.begin() + id);
    return S_OK;
}

CRCData* CRCWindowContainer::Get(int id)
{
    if (id < 0 || id >= data_.size()) return nullptr;

    CRCData* data = data_[id].get();
    return data;
}

int CRCWindowContainer::GetSize()
{
    return data_.size();
}

void CRCWindowContainer::Clear()
{
    data_.clear();
}