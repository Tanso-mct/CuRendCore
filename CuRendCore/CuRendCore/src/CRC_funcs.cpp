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

CRC_API std::unique_ptr<ICRCContainable> CRC::CreateWindowData(std::unique_ptr<CRCWindowSrc> attr)
{
    std::unique_ptr<CRCWindowAttr> windowData = std::make_unique<CRCWindowAttr>();

    windowData->src_ = std::move(attr);
    return windowData;
}

CRC_API std::unique_ptr<ICRCContainable> CRC::CreateSceneData(std::unique_ptr<CRCSceneSrc> attr)
{
    std::unique_ptr<CRCSceneAttr> sceneData = std::make_unique<CRCSceneAttr>();

    sceneData->src_ = std::move(attr);
    return sceneData;
}