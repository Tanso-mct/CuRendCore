#pragma once

// DLL export and import settings.
#define BUILDING_CRCBuffer_DLL
#ifdef BUILDING_CRCBuffer_DLL
#define CRC_BUFFER __declspec(dllexport)
#else
#define CRC_BUFFER __declspec(dllimport)
#endif

#pragma comment(lib, "cudart_static.lib")
#pragma comment(lib, "WinAppCore.lib")
#pragma comment(lib, "CudaCore.lib")