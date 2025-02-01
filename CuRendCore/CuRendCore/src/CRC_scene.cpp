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

HRESULT CRCSceneContainer::Remove(int id)
{
    if (id < 0 || id >= data_.size()) return E_FAIL;

    data_.erase(data_.begin() + id);
    return S_OK;
}

std::unique_ptr<CRCData> &CRCSceneContainer::Get(int id)
{
    if (id < 0 || id >= data_.size())
    {
        std::unique_ptr<CRCData> emptyData = nullptr;
        return emptyData;
    }
    
    return CRC::CastRef<CRCData>(data_[id]);
}

int CRCSceneContainer::GetSize()
{
    return data_.size();
}

void CRCSceneContainer::Clear()
{
    data_.clear();
}