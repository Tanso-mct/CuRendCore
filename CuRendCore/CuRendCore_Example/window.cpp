#include "window.h"

#include <iostream>

void MainWindowPhaseMethod::Awake()
{
    std::cout << "MainWindow Awake" << std::endl;
}

void MainWindowPhaseMethod::Update()
{
    std::cout << "MainWindow Update" << std::endl;
}

void MainWindowPhaseMethod::Show()
{
    std::cout << "MainWindow Show" << std::endl;
}

void MainWindowPhaseMethod::Hide()
{
    std::cout << "MainWindow Hide" << std::endl;
}

void MainWindowPhaseMethod::End()
{
    std::cout << "MainWindow End" << std::endl;
}
