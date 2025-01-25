#pragma once

// DLL export and import settings.
#define BUILDING_CRC_DLL
#ifdef BUILDING_CRC_DLL
#define CRC_API __declspec(dllexport)
#else
#define CRC_API __declspec(dllimport)
#endif
