#pragma once

#include "CuRendCore.h"

class MainWindowPhaseMethod : public ICRCPhaseMethod
{
public:
    MainWindowPhaseMethod() = default;
    ~MainWindowPhaseMethod() override = default;

    void Update() override;
    void Hide() override;
    void Restored() override;
    void End() override;
};