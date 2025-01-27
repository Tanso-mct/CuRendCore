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
    else return -1;
}

HRESULT CRCWindowContainer::Remove(int id)
{
    if (id < 0 || id >= data_.size()) return E_FAIL;

    data_.erase(data_.begin() + id);
    return S_OK;
}

std::unique_ptr<CRCData> &CRCWindowContainer::Get(int id)
{
    if (id < 0 || id >= data_.size())
    {
        std::unique_ptr<CRCData> emptyData = nullptr;
        return emptyData;
    }

    return CRC::CastRef<CRCData>(data_[id]);
}

int CRCWindowContainer::GetSize()
{
    return data_.size();
}

void CRCWindowContainer::Clear()
{
    data_.clear();
}