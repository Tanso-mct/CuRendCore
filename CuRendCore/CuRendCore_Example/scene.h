#pragma once

#include "CuRendCore.h"

class MainScenePhaseMethod : public ICRCPhaseMethod
{
public:
    MainScenePhaseMethod() = default;
    ~MainScenePhaseMethod() override = default;

    void Update() override;
    void Hide() override;
    void Restored() override;
    void End() override;
};