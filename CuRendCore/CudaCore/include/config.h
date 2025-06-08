#pragma once

// DLL export and import settings.
#define BUILDING_CudaCore_DLL
#ifdef BUILDING_CudaCore_DLL
#define CUDA_CORE __declspec(dllexport)
#else
#define CUDA_CORE __declspec(dllimport)
#endif

#pragma comment(lib, "WinAppCore.lib")
#pragma comment(lib, "cudart_static.lib")