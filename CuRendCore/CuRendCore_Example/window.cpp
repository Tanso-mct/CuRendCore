#include "window.h"

#include <iostream>

void MainWindowPhaseMethod::Update(ICRCContainable* attr)
{
    std::cout << "Main Window Update" << std::endl;
}

void MainWindowPhaseMethod::Hide(ICRCContainable* attr)
{
    std::cout << "Main Window Hide" << std::endl;
}

void MainWindowPhaseMethod::Restored(ICRCContainable* attr)
{
    std::cout << "Main Window Restored" << std::endl;
}

void MainWindowPhaseMethod::End(ICRCContainable* attr)
{
    std::cout << "Main Window End" << std::endl;
}
