#pragma once

// DLL export and import settings.
#define BUILDING_WACore_DLL
#ifdef BUILDING_WACore_DLL
#define WIN_APP_CORE __declspec(dllexport)
#else
#define WIN_APP_CORE __declspec(dllimport)
#endif