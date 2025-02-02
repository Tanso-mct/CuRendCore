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
    containers_.resize(CRC::CORE_CONTAINER_COUNT);
}

int CRCCore::Shutdown()
{
    CRC::Core() = nullptr;
    return 0;
}

HRESULT CRCCore::SetWindowContainer(std::unique_ptr<ICRCContainer> container)
{
    CRCWindowContainer* windowContainer = CRC::PtrAs<CRCWindowContainer>(container.get());

    if (windowContainer) containers_[CRC::ID_WINDOW_CONTAINER] = std::move(container);
    else return E_FAIL;
}

HRESULT CRCCore::SetSceneContainer(std::unique_ptr<ICRCContainer> container)
{
    CRCSceneContainer* sceneContainer = CRC::PtrAs<CRCSceneContainer>(container.get());

    if (sceneContainer) containers_[CRC::ID_SCENE_CONTAINER] = std::move(container);
    else return E_FAIL;
}

HRESULT CRCCore::CreateWindowCRC(int idWindow)
{
    if(
        containers_[CRC::ID_WINDOW_CONTAINER] == nullptr || 
        idWindow == CRC::ID_INVALID || 
        idWindow >= containers_[CRC::ID_WINDOW_CONTAINER]->GetSize()
    ) return E_FAIL;

    CRCWindowData* windowData = CRC::PtrAs<CRCWindowData>(containers_[CRC::ID_WINDOW_CONTAINER]->Get(idWindow));

    if (!RegisterClassEx(&windowData->src_->wcex_)) return E_FAIL;

    windowData->hWnd_ = CreateWindow
    (
        windowData->src_->wcex_.lpszClassName,
        windowData->src_->name_,
        windowData->src_->style_,
        windowData->src_->initialPosX_,
        windowData->src_->initialPosY_,
        windowData->src_->width_,
        windowData->src_->height_,
        windowData->src_->hWndParent_,
        nullptr,
        windowData->src_->hInstance,
        nullptr
    );
    if (!windowData->hWnd_) return E_FAIL;

    return S_OK;
}

HRESULT CRCCore::ShowWindowCRC(int idWindow)
{
    if(
        containers_[CRC::ID_WINDOW_CONTAINER] == nullptr || 
        idWindow == CRC::ID_INVALID || 
        idWindow >= containers_[CRC::ID_WINDOW_CONTAINER]->GetSize()
    ) return E_FAIL;

    CRCWindowData* windowData = CRC::PtrAs<CRCWindowData>(containers_[CRC::ID_WINDOW_CONTAINER]->Get(idWindow));

    if (!windowData->hWnd_) return E_FAIL;

    ShowWindow(windowData->hWnd_, SW_SHOW);
    UpdateWindow(windowData->hWnd_);

    return S_OK;
}

HRESULT CRCCore::CreateScene(int idScene)
{
    if(
        containers_[CRC::ID_SCENE_CONTAINER] == nullptr || 
        idScene == CRC::ID_INVALID || 
        idScene >= containers_[CRC::ID_SCENE_CONTAINER]->GetSize()
    ) return E_FAIL;

    //TODO: Implement scene creation.

    return S_OK;
}

HRESULT CRCCore::SetSceneToWindow(int idWindow, int idScene)
{
    if(
        containers_[CRC::ID_WINDOW_CONTAINER] == nullptr || containers_[CRC::ID_SCENE_CONTAINER] == nullptr ||
        idWindow == CRC::ID_INVALID || idScene == CRC::ID_INVALID ||
        idWindow >= containers_[CRC::ID_WINDOW_CONTAINER]->GetSize() || 
        idScene >= containers_[CRC::ID_SCENE_CONTAINER]->GetSize()
    ) return E_FAIL;

    CRCWindowData* windowData = CRC::PtrAs<CRCWindowData>(containers_[CRC::ID_WINDOW_CONTAINER]->Get(idWindow));
    windowData->idScene_ = idScene;

    return S_OK;
}