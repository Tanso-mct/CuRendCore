#include "CuRendCore/include/pch.h"
#include "CuRendCore/include/funcs.cuh"

#include "CuRendCore/include/scene.h"

CRCSceneAttr::CRCSceneAttr(std::string name)
: name(name)
{
}

std::unique_ptr<WACore::IContainable> CRCSceneFactory::Create(IDESC &desc) const
{
    CRC_SCENE_DESC* sceneDesc = WACore::As<CRC_SCENE_DESC>(&desc);
    if (!sceneDesc) return nullptr;

    std::unique_ptr<CRCSceneAttr> sceneAttr = std::make_unique<CRCSceneAttr>(sceneDesc->name_);
    return sceneAttr;
}
