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

namespace CRC
{

constexpr int ERROR_CREATE_WINDOW = 0x0001;
constexpr int ERROR_SHOW_WINDOW = 0x0002;
constexpr int ERROR_CREATE_SCENE = 0x0003;
constexpr int ERROR_CREATE_CONTAINER = 0x0004;
constexpr int ERROR_ADD_LISTENER = 0x0005;

}