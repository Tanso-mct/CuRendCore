#include "window.h"

#include <iostream>

void MainWindowListener::OnUpdate(UINT msg, WPARAM wParam, LPARAM lParam)
{
    // std::cout << "Main Window Update" << std::endl;
}

void MainWindowListener::OnSize(UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "Main Window Size" << std::endl;
}

void MainWindowListener::OnDestroy(UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "Main Window Destroy" << std::endl;
    PostQuitMessage(0);
}
