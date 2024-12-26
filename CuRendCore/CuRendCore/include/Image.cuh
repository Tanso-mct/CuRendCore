#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CRCConfig.h"

#include "UI.cuh"

namespace CRC
{

class Image : public UI
{
private:
    Image(UI_ATTR& uiattr) : UI(uiattr) { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };
public:
    ~Image() override { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };


    friend class UIFactory;
};



} // namespace CRC
