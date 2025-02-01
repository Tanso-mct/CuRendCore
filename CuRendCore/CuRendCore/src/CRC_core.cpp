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

HRESULT CRCCore::SetSceneToWindow(int idWindow, int idScene)
{
    if (windowContainer_ == nullptr || sceneContainer_ == nullptr) return E_FAIL;
    if (idWindow == CRC::INVALID_ID || idScene == CRC::INVALID_ID) return E_FAIL;
    if (idWindow >= windowContainer_->GetSize() || idScene >= sceneContainer_->GetSize()) return E_FAIL;

    std::unique_ptr<CRCWindowData>& windowData = CRC::CastRef<CRCWindowData>(windowContainer_->Get(idWindow));

    return S_OK;
}