#include "scene.h"

#include <iostream>

void MainScenePhaseMethod::Awake()
{
    std::cout << "Main Scene Awake" << std::endl;
}

void MainScenePhaseMethod::Update()
{
    std::cout << "Main Scene Update" << std::endl;
}

void MainScenePhaseMethod::Show()
{
    std::cout << "Main Scene Show" << std::endl;
}

void MainScenePhaseMethod::Hide()
{
    std::cout << "Main Scene Hide" << std::endl;
}

void MainScenePhaseMethod::End()
{
    std::cout << "Main Scene End" << std::endl;
}
