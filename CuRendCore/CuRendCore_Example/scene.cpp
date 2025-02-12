#include "scene.h"

#include <iostream>

void MainScenePhaseMethod::Update(ICRCContainable* attr)
{
    std::cout << "Main Scene Update" << std::endl;
}

void MainScenePhaseMethod::Hide(ICRCContainable* attr)
{
    std::cout << "Main Scene Hide" << std::endl;
}

void MainScenePhaseMethod::Restored(ICRCContainable* attr)
{
    std::cout << "Main Scene Restored" << std::endl;
}

void MainScenePhaseMethod::End(ICRCContainable* attr)
{
    std::cout << "Main Scene End" << std::endl;
}
