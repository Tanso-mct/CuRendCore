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

constexpr const char* C_COLOR_MSG = "\033[36m";
constexpr const char* C_COLOR_ERROR = "\033[31m";
constexpr const char* C_COLOR_WARNING = "\033[33m";

constexpr const char* C_COLOR_RESET = "\033[0m";

constexpr const char* C_TAG = "[CuRendCore]";
constexpr const char* C_TAG_END = "[------ END]";

constexpr int ERROR_CREATE_WINDOW = 0x0001;
constexpr int ERROR_SHOW_WINDOW = 0x0002;
constexpr int ERROR_CREATE_SCENE = 0x0003;
constexpr int ERROR_CREATE_CONTAINER = 0x0004;
constexpr int ERROR_ADD_TO_CONTAINER = 0x0005;
constexpr int ERROR_CAST = 0x0006;


} // namespace CRC

enum class CRC_API CRC_FEATURE_LEVEL : unsigned int
{
    L0_0 = 0,
};

enum class CRC_API CRC_RENDER_MODE : unsigned int
{
    CUDA = 0,
    D3D11 = 1,
};