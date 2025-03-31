#include "window.h"

#include <iostream>

MainWindowEvent::MainWindowEvent(int& idMainWindow, int& idMainScene, int& idUserInput)
: idMainWindow_(idMainWindow), idMainScene_(idMainScene), idUserInput_(idUserInput)
{
}

MainWindowEvent::~MainWindowEvent()
{
}

void MainWindowEvent::OnUpdate(std::unique_ptr<WACore::IContainer> &container, UINT msg, WPARAM wParam, LPARAM lParam)
{
    // std::cout << "Main Window Update" << std::endl;
}

void MainWindowEvent::OnSize(std::unique_ptr<WACore::IContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "Main Window Size" << std::endl;
}

void MainWindowEvent::OnDestroy(std::unique_ptr<WACore::IContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "Main Window Destroy" << std::endl;
    PostQuitMessage(0);
}
