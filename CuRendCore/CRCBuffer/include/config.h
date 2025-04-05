#pragma once

// DLL export and import settings.
#define BUILDING_CRCBuffer_DLL
#ifdef BUILDING_CRCBuffer_DLL
#define CRC_BUFFER __declspec(dllexport)
#else
#define CRC_BUFFER __declspec(dllimport)
#endif