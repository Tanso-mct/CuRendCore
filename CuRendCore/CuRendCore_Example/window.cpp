#include "window.h"

#include <iostream>

void MainWindowListener::OnUpdate(ICRCContainable *attr, UINT msg, WPARAM wParam, LPARAM lParam)
{
    // std::cout << "Main Window Update" << std::endl;
}

void MainWindowListener::OnMinimize(ICRCContainable *attr, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "Main Window Minimize" << std::endl;
}

void MainWindowListener::OnMaximize(ICRCContainable *attr, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "Main Window Maximize" << std::endl;
}

void MainWindowListener::OnRestored(ICRCContainable *attr, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "Main Window Restored" << std::endl;
}

void MainWindowListener::OnDestroy(ICRCContainable *attr, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "Main Window Destroy" << std::endl;
    PostQuitMessage(0);
}
