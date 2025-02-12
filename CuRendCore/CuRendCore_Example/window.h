#pragma once

#include "CuRendCore.h"

class MainWindowPhaseMethod : public ICRCPhaseMethod
{
public:
    MainWindowPhaseMethod() = default;
    ~MainWindowPhaseMethod() override = default;

    void Update(ICRCContainable* attr) override;
    void Hide(ICRCContainable* attr) override;
    void Restored(ICRCContainable* attr) override;
    void End(ICRCContainable* attr) override;
};