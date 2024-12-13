#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CRCConfig.h"

#include "Object.cuh"

namespace CRC
{

class FromObj : public Object
{
private:
    FromObj(OBJECT_ATTR& oattr) : Object(oattr) { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };
public:
    ~FromObj() override { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };


    friend class ObjectFactory;
};



} // namespace CRC
