#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CRCConfig.h"

#include "Utility.cuh"
#include "Math.cuh"

namespace CRC
{

class Camera : public Utility
{
private:
    Camera(UTILITY_ATTR& utattr);;

    Vec3d lookAt;
    float nearZ = 0;
    float farZ = 0;
    float fov = 0;
    Vec2d aspectRatio;

    void SetViewMatrix();
    void SetViewVolume();

public:
    ~Camera() override { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };


    friend class UtilityFactory;
};



} // namespace CRC