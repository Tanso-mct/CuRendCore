#include "test.h"

int WINAPI WinMain
(
    const HINSTANCE hInstance, const HINSTANCE hPrevInstance,
    const LPSTR lpCmdLine, const int nCmdShow
) {
    CRC::ExampleClass example;
    example.exampleMethod();
    return 0;
}