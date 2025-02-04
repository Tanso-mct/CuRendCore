#include "window.h"

#include <iostream>

void MainWindowPhaseMethod::Awake()
{
    std::cout << "Main Window Awake" << std::endl;
}

void MainWindowPhaseMethod::Update()
{
    std::cout << "Main Window Update" << std::endl;
}

void MainWindowPhaseMethod::Show()
{
    std::cout << "Main Window Show" << std::endl;
}

void MainWindowPhaseMethod::Hide()
{
    std::cout << "Main Window Hide" << std::endl;
}

void MainWindowPhaseMethod::End()
{
    std::cout << "Main Window End" << std::endl;
}
