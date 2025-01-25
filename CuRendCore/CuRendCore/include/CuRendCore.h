#pragma once

#include "CRC_config.h"
#include "CRC_core.h"
#include "CRC_interface.h"

namespace CRC
{

std::unique_ptr<CRCCore> CRC_API CreateCRCCore();

std::unique_ptr<CRCContainer> CRC_API CreateWindowContainer();
std::unique_ptr<CRCContainer> CRC_API CreateSceneContainer();

std::unique_ptr<CRCData> CRC_API CreateWindowData();
std::unique_ptr<CRCData> CRC_API CreateSceneData();

}