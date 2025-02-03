#pragma once

#include "CuRendCore.h"

class MainScenePhaseMethod : public ICRCPhaseMethod
{
public:
    MainScenePhaseMethod() = default;
    ~MainScenePhaseMethod() override = default;

    void Awake() override;
    void Start() override;
    void Update() override;
    void End() override;
};