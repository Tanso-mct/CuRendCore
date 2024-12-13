#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CRCConfig.h"

#include "Utility.cuh"
#include "Math.cuh"

namespace CRC
{

typedef struct CRC_API _CAMERA_DEVICE_DATA
{
    Matrix mtView;
    Vec3d viewVolumeVs[8];
    float nearZ;
    float farZ;
    float fovXzCos = 0;
    float fovYzCos = 0;
} CAMERA_DDATA;

class Camera : public Utility
{
private:
    Camera(UTILITYATTR& utattr);;

    Vec3d eye;
    Vec3d at;

    float fov = 0;
    Vec2d aspectRatio;

    // Device data.
    CAMERA_DDATA* ddata = nullptr;

    void SetViewMatrix();
    void SetViewVolume();

public:
    ~Camera() override { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };


    friend class UtilityFactory;
};



} // namespace CRC