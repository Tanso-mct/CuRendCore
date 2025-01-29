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

std::unique_ptr<CRCData> CRCWindowContainer::Take(int id)
{
    if (id < 0 || id >= data_.size()) return nullptr;

    return std::move(data_[id]);
}

HRESULT CRCWindowContainer::Set(int id, std::unique_ptr<CRCData> &data)
{
    if (id < 0 || id >= data_.size()) return E_FAIL;
    
    data_[id] = std::move(CRC::CastMove<CRCWindowData>(data));
}

UINT CRCWindowContainer::GetSize()
{
    return data_.size();
}

void CRCWindowContainer::Clear(int id)
{
    if (id < 0 || id >= data_.size()) return;

    data_[id] = nullptr;
}

void CRCWindowContainer::Clear()
{
    data_.clear();
}