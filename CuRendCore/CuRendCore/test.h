#pragma once

#define BUILDING_CRC_DLL

#ifdef BUILDING_CRC_DLL
#define CRC_API __declspec(dllexport)
#else
#define CRC_API __declspec(dllimport)
#endif

#include <Windows.h>

namespace CRC 
{

class CRC_API ExampleClass {
public:
    ExampleClass();
    void exampleMethod();
};

}