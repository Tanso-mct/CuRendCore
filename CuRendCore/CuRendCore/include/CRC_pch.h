﻿#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <Windows.h>
#include <string>
#include <memory>
#include <vector>

namespace CRC
{

constexpr int ERROR_CREATE_WINDOW = 0x0001;
constexpr int ERROR_SHOW_WINDOW = 0x0002;
constexpr int ERROR_CREATE_SCENE = 0x0003;
constexpr int ERROR_SET_SCENE_TO_WINDOW = 0x0004;

constexpr int CORE_CONTAINER_COUNT = 4;
constexpr int ID_WINDOW_CONTAINER = 0;
constexpr int ID_SCENE_CONTAINER = 1;
constexpr int ID_WINDOW_PM_CONTAINER = 2;
constexpr int ID_SCENE_PM_CONTAINER = 3;


}