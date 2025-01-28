#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <Windows.h>
#include <string>
#include <memory>

namespace CRC
{

constexpr UINT ERROR_CREATE_WINDOW = 0x0001;
constexpr UINT ERROR_SHOW_WINDOW = 0x0002;
constexpr UINT ERROR_CREATE_SCENE = 0x0003;

constexpr int INVALID_ID = -1;

}