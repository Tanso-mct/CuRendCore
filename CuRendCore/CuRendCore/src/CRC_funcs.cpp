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

CRC_API std::unique_ptr<CRCData> CRC::CreateWindowData(std::unique_ptr<CRCWindowAttr> &attr)
{
    std::unique_ptr<CRCWindowData> windowData = std::make_unique<CRCWindowData>();

    windowData->src_ = std::move(attr);
    return windowData;
}

CRC_API HRESULT CRC::CreateCRCWindow(int id)
{
    std::unique_ptr<CRCWindowData> windowData = CRC::GetAs<CRCWindowData>(CRC::Core()->WindowContainer()->Take(id));




    if (!RegisterClassEx(&datas[i]->src_->wcex_))
    {
        thread.error = CRC::THRD_ERROR_FAIL;
        break;
    }

    datas[i]->hWnd_ = CreateWindow
    (
        datas[i]->src_->wcex_.lpszClassName,
        datas[i]->src_->name_,
        datas[i]->src_->style_,
        datas[i]->src_->initialPosX_,
        datas[i]->src_->initialPosY_,
        datas[i]->src_->width_,
        datas[i]->src_->height_,
        datas[i]->src_->hWndParent_,
        nullptr,
        datas[i]->src_->hInstance,
        nullptr
    );

    if (!datas[i]->hWnd_)
    {
        thread.error = CRC::THRD_ERROR_FAIL;
        break;
    }

    // The source data is no longer needed, so make it nullptr
    datas[i]->src_ = nullptr;

    // Because it was created, set the flag to False.
    datas[i]->needCreateFlag_ = false;
}

CRC_API std::unique_ptr<CRCData> CRC::CreateSceneData(std::unique_ptr<CRCSceneAttr> &attr)
{
    std::unique_ptr<CRCSceneData> sceneData = std::make_unique<CRCSceneData>();

    sceneData->needCreateFlag_ = attr->needCreateFlag_;
    sceneData->src_ = std::move(attr);

    return sceneData;
}

CRC_API void CRC::CreateSceneAsync(int id)
{
    CRC::Core()->SendCmdToSceneThrd(CRC::THRD_CMD_CREATE_SCENES);
}