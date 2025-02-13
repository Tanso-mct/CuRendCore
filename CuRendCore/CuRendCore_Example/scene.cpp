#include "scene.h"

#include <iostream>

void MainSceneListener::OnUpdate(ICRCContainable *attr, UINT msg, WPARAM wParam, LPARAM lParam)
{
    // std::cout << "Main Scene Update" << std::endl;
}

void MainSceneListener::OnSize(ICRCContainable *attr, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "Main Scene Size" << std::endl;
}

void MainSceneListener::OnDestroy(ICRCContainable *attr, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "Main Scene End" << std::endl;
}
