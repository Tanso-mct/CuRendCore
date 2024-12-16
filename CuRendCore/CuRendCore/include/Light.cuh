#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CRCConfig.h"

#include "Utility.cuh"

namespace CRC
{

class Light : public Utility
{
private:
    Light(UTILITY_ATTR& utattr) : Utility(utattr) { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };
public:
    ~Light() override { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };


    friend class UtilityFactory;
};



} // namespace CRC
