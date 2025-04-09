#pragma once

// DLL export and import settings.
#define BUILDING_CRCTexture_DLL
#ifdef BUILDING_CRCTexture_DLL
#define CRC_TEXTURE __declspec(dllexport)
#else
#define CRC_TEXTURE __declspec(dllimport)
#endif