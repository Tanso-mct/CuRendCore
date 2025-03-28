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
#include <stdexcept>
#include <cassert>
#include <cstddef>
#include <string_view>
#include <initializer_list>
#include <thread>

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "packages/WinAppCore/include/WACore.h"

#include "CuRendCore/include/data_cast.h"