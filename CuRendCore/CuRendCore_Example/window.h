#pragma once

#include "CuRendCore.h"

class MainWindowPhaseMethod : public ICRCPhaseMethod
{
public:
    MainWindowPhaseMethod() = default;
    ~MainWindowPhaseMethod() override = default;

    void Awake() override;
    void Start() override;
    void Update() override;
    void End() override;
};