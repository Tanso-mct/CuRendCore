#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_scene.h"

CRCSceneAttr::CRCSceneAttr(std::string name)
: name(name)
{
}

std::unique_ptr<ICRCContainable> CRCSceneFactory::Create(IDESC &desc) const
{
    CRC_SCENE_DESC* sceneDesc = CRC::As<CRC_SCENE_DESC>(&desc);
    if (!sceneDesc) return nullptr;

    std::unique_ptr<CRCSceneAttr> sceneAttr = std::make_unique<CRCSceneAttr>(sceneDesc->name_);
    return sceneAttr;
}
