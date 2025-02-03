#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_scene.h"

int CRCSceneContainer::Add(std::unique_ptr<ICRCContainable>data)
{
    std::unique_ptr<CRCSceneAttr> sceneData = CRC::UniqueAs<CRCSceneAttr>(data);

    if (sceneData)
    {
        data_.push_back(std::move(sceneData));
        return data_.size() - 1;
    }
    else return CRC::ID_INVALID;
}

HRESULT CRCSceneContainer::Remove(int id)
{
    if (id < 0 || id >= data_.size()) return E_FAIL;

    data_[id].reset();
    return S_OK;
}

ICRCContainable* CRCSceneContainer::Get(int id)
{
    if (id < 0 || id >= data_.size()) return nullptr;
    
    return CRC::PtrAs<ICRCContainable>(data_[id].get());
}

int CRCSceneContainer::GetSize()
{
    return data_.size();
}

void CRCSceneContainer::Clear()
{
    data_.clear();
}