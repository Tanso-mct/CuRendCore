#pragma once

#include "CuRendCore.h"

class MainScenePhaseMethod : public ICRCPhaseMethod
{
public:
    MainScenePhaseMethod() = default;
    ~MainScenePhaseMethod() override = default;

    void Update(ICRCContainable* attr) override;
    void Hide(ICRCContainable* attr) override;
    void Restored(ICRCContainable* attr) override;
    void End(ICRCContainable* attr) override;
};