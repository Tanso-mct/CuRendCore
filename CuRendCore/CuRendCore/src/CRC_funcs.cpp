#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_core.h"
#include "CRC_window.h"
#include "CRC_scene.h"

CRC_API std::unique_ptr<CRCCore>& CRC::Core()
{
    static std::unique_ptr<CRCCore> core = std::make_unique<CRCCore>();
    return core;
}

CRC_API std::unique_ptr<ICRCData> CRC::CreateWindowData(std::unique_ptr<CRCWindowAttr> attr)
{
    std::unique_ptr<CRCWindowData> windowData = std::make_unique<CRCWindowData>();

    windowData->src_ = std::move(attr);
    return windowData;
}

CRC_API std::unique_ptr<ICRCData> CRC::CreateSceneData(std::unique_ptr<CRCSceneAttr> attr)
{
    std::unique_ptr<CRCSceneData> sceneData = std::make_unique<CRCSceneData>();

    sceneData->src_ = std::move(attr);
    return sceneData;
}