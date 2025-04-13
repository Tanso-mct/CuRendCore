#pragma once

// DLL export and import settings.
#define BUILDING_CRCTexture_DLL
#ifdef BUILDING_CRCTexture_DLL
#define CRC_TEXTURE __declspec(dllexport)
#else
#define CRC_TEXTURE __declspec(dllimport)
#endif

#pragma comment(lib, "cudart_static.lib")
#pragma comment(lib, "WinAppCore.lib")
#pragma comment(lib, "CudaCore.lib")