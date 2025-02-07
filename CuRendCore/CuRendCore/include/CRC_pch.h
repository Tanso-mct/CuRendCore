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

namespace CRC
{

constexpr int ERROR_CREATE_WINDOW = 0x0001;
constexpr int ERROR_SHOW_WINDOW = 0x0002;
constexpr int ERROR_CREATE_SCENE = 0x0003;
constexpr int ERROR_SET_SCENE_TO_WINDOW = 0x0004;
constexpr int ERROR_CREATE_CONTAINER = 0x0005;
constexpr int ERROR_CREATE_PM = 0x0006;
constexpr int ERROR_ADD_PM_TO_WINDOW = 0x0007;

}