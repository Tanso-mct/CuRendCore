#pragma once

#include "CRC_config.h"
#include "CRC_container.h"
#include "CRC_factory.h"

#include <Windows.h>
#include <memory>
#include <vector>
#include <string>

struct CRC_API CRC_SCENE_DESC : public IDESC
{
    std::string name_ = "Scene";
};

class CRC_API CRCSceneFactory : public ICRCFactory
{
public:
    virtual ~CRCSceneFactory() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCSceneAttr : public ICRCContainable
{
private:
    const std::string name;
public:
    CRCSceneAttr(std::string name = "Scene");
    virtual ~CRCSceneAttr() override = default;

    const std::string& GetName() const { return name; }

};