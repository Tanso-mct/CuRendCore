#pragma once

#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma comment(lib, "cudart_static.lib")

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "WinAppCore/include/WACore.h"
#pragma comment(lib, "WinAppCore.lib")

#include "CudaCore/include/CudaCore.h"
#pragma comment(lib, "CudaCore.lib")

#include "CRCTexture/include/console.h"