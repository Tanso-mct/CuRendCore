#include "CRC_pch.h"

#include "CRC_funcs.h"

#include "CRC_core.h"
#include "CRC_window.h"

std::unique_ptr<CRCCore> CRC_API CRC::CreateCRCCore()
{
    return std::make_unique<CRCCore>();
}

std::unique_ptr<CRCContainer> CRC_API CRC::CreateWindowContainer()
{
    std::unique_ptr<CRCContainer> windowContainer = std::make_unique<CRCWindowContainer>();
    return windowContainer;
}

std::unique_ptr<CRCData> CRC_API CRC::CreateWindowData(CRCWindowAttr& attr)
{
    std::unique_ptr<CRCWindowData> windowData = std::make_unique<CRCWindowData>();

    // Set windowData properties.

    return windowData;
}
