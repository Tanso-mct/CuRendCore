#pragma once

// DLL export and import settings.
#define BUILDING_CRCDevice_DLL
#ifdef BUILDING_CRCDevice_DLL
#define CRC_DEVICE __declspec(dllexport)
#else
#define CRC_DEVICE __declspec(dllimport)
#endif

#pragma comment(lib, "WinAppCore.lib")