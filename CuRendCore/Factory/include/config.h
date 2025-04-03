#pragma once

// DLL export and import settings.
#define BUILDING_FACTORY_DLL
#ifdef BUILDING_FACTORY_DLL
#define FACTORY_API __declspec(dllexport)
#else
#define FACTORY_API __declspec(dllimport)
#endif