#pragma once

#include "CRC_config.h"
#include "CRC_container.h"

class CRC_API ICRCPhaseMethod : public ICRCContainable
{
public:
    virtual ~ICRCPhaseMethod() = default;

    virtual void Update() = 0;
    virtual void Hide() = 0;
    virtual void Restored() = 0;
    virtual void End() = 0;
};