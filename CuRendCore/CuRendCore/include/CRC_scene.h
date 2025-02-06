#pragma once

#include "CRC_config.h"
#include "CRC_interface.h"

#include <Windows.h>
#include <memory>
#include <vector>
#include <string>

struct CRC_API CRCSceneSrc
{
    std::string name_ = "Scene";
};

class CRC_API CRCSceneAttr : public ICRCContainable
{
public:
    virtual ~CRCSceneAttr() override = default;
    std::unique_ptr<CRCSceneSrc> src_ = nullptr;

    bool isAwaked_ = false;
    ICRCPhaseMethod* phaseMethod_ = nullptr;
};