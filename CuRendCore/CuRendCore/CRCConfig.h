#pragma once

#define BUILDING_CRC_DLL

#ifdef BUILDING_CRC_DLL
#define CRC_API __declspec(dllexport)
#else
#define CRC_API __declspec(dllimport)
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN // Extract only the essentials from the large number of descriptions in Windows.h.
#endif

#include <Windows.h>

#define CRC_SLOT unsigned int
#define CRC_SLOT_INVALID -1

#define CRC_WND_DEFAULT_NAME L"CuRendCore Window"
#define CRC_WND_DEFAULT_POS_X CW_USEDEFAULT
#define CRC_WND_DEFAULT_POS_Y CW_USEDEFAULT
#define CRC_WND_DEFAULT_WIDTH 800
#define CRC_WND_DEFAULT_HEIGHT 600
#define CRC_WND_DEFAULT_STYLE WS_OVERLAPPEDWINDOW