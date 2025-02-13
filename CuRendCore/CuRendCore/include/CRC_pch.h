#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <Windows.h>
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <utility>
#include <iostream>
#include <chrono>
#include <ctime>

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace CRC
{

constexpr int ERROR_CREATE_WINDOW = 0x0001;
constexpr int ERROR_SHOW_WINDOW = 0x0002;
constexpr int ERROR_CREATE_SCENE = 0x0003;
constexpr int ERROR_CREATE_CONTAINER = 0x0004;
constexpr int ERROR_ADD_LISTENER = 0x0005;

}