#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <Windows.h>
#include <string>
#include <memory>
#include <thread>
#include <atomic>

namespace CRC
{

constexpr int INVALID_ID = -1;

constexpr UINT ERROR_CREATE_WINDOW = 0x0001;
constexpr UINT ERROR_SHOW_WINDOW = 0x0002;
constexpr UINT ERROR_CREATE_SCENE = 0x0003;
constexpr UINT ERROR_SET_SCENE_TO_WINDOW = 0x0004;

constexpr int THRD_CMD_EXIT = 0x0000;
constexpr int THRD_CMD_READY = 0x0001;
constexpr int THRD_CMD_CREATE_WINDOWS = 0x0002;
constexpr int THRD_CMD_CREATE_SCENES = 0x0003;

constexpr int THRD_ERROR_OK = 0x0000;
constexpr int THRD_ERROR_FAIL = 0x0001;
constexpr int THRD_ERROR_NOT_FOUND_CMD = 0x0002;

}