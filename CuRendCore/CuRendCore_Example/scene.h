#pragma once

#include "CuRendCore.h"

class MainScenePhaseMethod : public ICRCPhaseMethod
{
public:
    MainScenePhaseMethod() = default;
    ~MainScenePhaseMethod() override = default;

    void Awake() override;
    void Update() override;
    void Show() override;
    void Hide() override;
    void End() override;
};