#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_scene.h"

int CRCSceneContainer::Add(std::unique_ptr<CRCData> &data)
{
    std::unique_ptr<CRCSceneData> sceneData = CRC::CastMove<CRCSceneData>(data);

    if (sceneData)
    {
        data_.push_back(std::move(sceneData));
        return data_.size() - 1;
    }
    else return CRC::INVALID_ID;
}

std::unique_ptr<CRCData> CRCSceneContainer::Take(int id)
{
    if (id < 0 || id >= data_.size()) return nullptr;
    
    return std::move(data_[id]);
}

HRESULT CRCSceneContainer::Set(int id, std::unique_ptr<CRCData> &data)
{
    if (id < 0 || id >= data_.size()) return E_FAIL;

    data_[id] = CRC::CastMove<CRCSceneData>(data);
}

UINT CRCSceneContainer::GetSize()
{
    return data_.size();
}

void CRCSceneContainer::Clear(int id)
{
    if (id < 0 || id >= data_.size()) return;

    data_[id] = nullptr;
}

void CRCSceneContainer::Clear()
{
    data_.clear();
}
