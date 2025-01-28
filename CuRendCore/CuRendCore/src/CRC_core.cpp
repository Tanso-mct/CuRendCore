#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_core.h"
#include "CRC_window.h"
#include "CRC_scene.h"

CRCCore::CRCCore()
{
}

CRCCore::~CRCCore()
{
}

void CRCCore::Initialize()
{
}

void CRCCore::Run()
{
}

int CRCCore::Shutdown()
{
    CRC::Core() = nullptr;
    return 0;
}

HRESULT CRCCore::MoveWindowContainer(std::unique_ptr<CRCContainer>& container)
{
    std::unique_ptr<CRCWindowContainer> windowContainer = CRC::CastMove<CRCWindowContainer>(container);

    if (windowContainer)
    {
        windowContainer_ = std::move(windowContainer);
        return S_OK;
    }
    else return E_FAIL;
}

HRESULT CRCCore::MoveSceneContainer(std::unique_ptr<CRCContainer> &container)
{
    std::unique_ptr<CRCSceneContainer> sceneContainer = CRC::CastMove<CRCSceneContainer>(container);

    if (sceneContainer)
    {
        sceneContainer_ = std::move(sceneContainer);
        return S_OK;
    }
    else return E_FAIL;
}
