#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

// DLL export and import settings.
#define BUILDING_CRC_DLL
#ifdef BUILDING_CRC_DLL
#define CRC_API __declspec(dllexport)
#else
#define CRC_API __declspec(dllimport)
#endif

namespace CRC
{

constexpr int ID_INVALID = -1;

constexpr const char* C_TAG_COLOR = "\033[36m";
constexpr const char* C_COLOR_RESET = "\033[0m";

constexpr const char* C_TAG = "[CuRendCore]";

}