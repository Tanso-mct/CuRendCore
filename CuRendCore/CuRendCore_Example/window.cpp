#include "window.h"

#include <iostream>

void MainWindowPhaseMethod::Update()
{
    std::cout << "Main Window Update" << std::endl;
}

void MainWindowPhaseMethod::Hide()
{
    std::cout << "Main Window Hide" << std::endl;
}

void MainWindowPhaseMethod::Restored()
{
    std::cout << "Main Window Restored" << std::endl;
}

void MainWindowPhaseMethod::End()
{
    std::cout << "Main Window End" << std::endl;
}
