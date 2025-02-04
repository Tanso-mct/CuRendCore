#pragma once

#include "CuRendCore.h"

class MainWindowPhaseMethod : public ICRCPhaseMethod
{
public:
    MainWindowPhaseMethod() = default;
    ~MainWindowPhaseMethod() override = default;

    void Awake() override;
    void Update() override;
    void Show() override;
    void Hide() override;
    void End() override;
};