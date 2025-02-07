#include "scene.h"

#include <iostream>

void MainScenePhaseMethod::Update()
{
    std::cout << "Main Scene Update" << std::endl;
}

void MainScenePhaseMethod::Hide()
{
    std::cout << "Main Scene Hide" << std::endl;
}

void MainScenePhaseMethod::Restored()
{
    std::cout << "Main Scene Restored" << std::endl;
}

void MainScenePhaseMethod::End()
{
    std::cout << "Main Scene End" << std::endl;
}
