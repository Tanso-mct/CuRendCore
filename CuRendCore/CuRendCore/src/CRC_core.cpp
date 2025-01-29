#include "CRC_pch.h"
#include "CRC_funcs.h"

#include <mutex>

#include "CRC_core.h"
#include "CRC_window.h"
#include "CRC_scene.h"

void CRCCore::Initialize()
{
    // Start window thread.
    CRC::Core()->threadWindow_.thread_ = std::thread(CRCCore::WindowThread, std::ref(threadWindow_));
    CRC::Core()->threadWindow_.isRunning_ = true;

    // Start scene thread.
    CRC::Core()->threadScene_.thread_ = std::thread(CRCCore::SceneThread, std::ref(threadScene_));
    CRC::Core()->threadScene_.isRunning_ = true;
}

int CRCCore::Shutdown()
{
    // Stop window thread.
    CRC::Core()->threadWindow_.isRunning_ = false;
    CRC::Core()->threadWindow_.thread_.join();

    // Stop scene thread.
    CRC::Core()->threadScene_.isRunning_ = false;
    CRC::Core()->threadScene_.thread_.join();

    CRC::Core() = nullptr;
    return 0;
}

std::unique_ptr<CRCContainer>& CRCCore::WindowContainer()
{
    static std::unique_ptr<CRCWindowContainer> windowContainer_ = std::make_unique<CRCWindowContainer>();

    std::lock_guard<std::mutex> lock(windowContainer_->mtx);
    return CRC::CastRef<CRCContainer>(windowContainer_);
}

std::unique_ptr<CRCContainer> &CRCCore::SceneContainer()
{
    static std::unique_ptr<CRCSceneContainer> sceneContainer_ = std::make_unique<CRCSceneContainer>();

    std::lock_guard<std::mutex> lock(sceneContainer_->mtx);
    return CRC::CastRef<CRCContainer>(sceneContainer_);
}

void CRCCore::SendCmdToWindowThrd(int cmd)
{
    threadWindow_.isCmd = cmd;
}

int CRCCore::ReceiveCmdFromWindowThrd()
{
    return threadWindow_.isCmd;
}

void CRCCore::GetResultFromWindowThrd(int &didCmd, int &error)
{
    didCmd = threadWindow_.didCmd;
    error = threadWindow_.error;
}

void CRCCore::SendCmdToSceneThrd(int cmd)
{
    threadScene_.isCmd = cmd;
}

int CRCCore::ReceiveCmdFromSceneThrd()
{
    return threadScene_.isCmd;
}

void CRCCore::GetResultFromSceneThrd(int &didCmd, int &error)
{
    didCmd = threadScene_.didCmd;
    error = threadScene_.error;
}

void CRCCore::WindowThread(CRCThread& thread)
{
    while (thread.isRunning_)
    {
        switch (thread.isCmd)
        {
        case CRC::THRD_CMD_READY:
            break;

        case CRC::THRD_CMD_CREATE_WINDOWS:
        {
            std::vector<std::unique_ptr<CRCWindowData>> datas;
            {
                std::unique_ptr<CRCContainer>& container = CRC::Core()->WindowContainer();
                std::lock_guard<std::mutex> lock(container->mtx);

                for (int i = 0; i < container->GetSize(); i++)
                {
                    datas.push_back(CRC::GetAs<CRCWindowData>(container->Take(i)));
                }
            }

            for (int i = 0; i < datas.size(); i++)
            {
                // If the window needs to be created, create it.
                if (datas[i]->needCreateFlag_ == false) continue;

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

            {
                std::unique_ptr<CRCContainer>& container = CRC::Core()->WindowContainer();
                std::lock_guard<std::mutex> lock(container->mtx);

                for (int i = 0; i < datas.size(); i++)
                {
                    container->Set(i, CRC::CastRef<CRCData>(datas[i]));
                }
            }

            thread.didCmd = CRC::THRD_CMD_CREATE_WINDOWS;
            thread.isCmd = CRC::THRD_CMD_EXIT;
            break;
        }

        default:
            thread.error = CRC::THRD_ERROR_NOT_FOUND_CMD;
            thread.isCmd = CRC::THRD_CMD_EXIT;
            break;
        }

        thread.error = CRC::THRD_ERROR_OK;
    }
}

void CRCCore::SceneThread(CRCThread& thread)
{
    std::mutex mtx;
    while (thread.isRunning_)
    {
        switch (thread.isCmd)
        {
        case CRC::THRD_CMD_READY:
            break;

        case CRC::THRD_CMD_CREATE_SCENES:
        {
            std::vector<std::unique_ptr<CRCSceneData>> datas;
            {
                std::unique_ptr<CRCContainer>& container = CRC::Core()->SceneContainer();
                std::lock_guard<std::mutex> lock(container->mtx);

                for (int i = 0; i < container->GetSize(); i++)
                {
                    datas.push_back(CRC::GetAs<CRCSceneData>(container->Take(i)));
                }
            }

            for (int i = 0; i < datas.size(); i++)
            {
                // If the scene needs to be created, create it.
                if (datas[i]->needCreateFlag_ == false) continue;

                //TODO: Create scene from scene attributes here.


                // The source data is no longer needed, so make it nullptr             
                datas[i]->src_ = nullptr;

                // Because it was created, set the flag to False.
                datas[i]->needCreateFlag_ = false;
            }

            {
                std::unique_ptr<CRCContainer>& container = CRC::Core()->SceneContainer();
                std::lock_guard<std::mutex> lock(container->mtx);

                for (int i = 0; i < datas.size(); i++)
                {
                    container->Set(i, CRC::CastRef<CRCData>(datas[i]));
                }
            }

            thread.didCmd = CRC::THRD_CMD_CREATE_SCENES;
            thread.isCmd = CRC::THRD_CMD_EXIT;
            break;
        }

            
        default:
            thread.error = CRC::THRD_ERROR_NOT_FOUND_CMD;
            thread.isCmd = CRC::THRD_CMD_EXIT;
            break;
        }

        thread.error = CRC::THRD_ERROR_OK;
    }
}

HRESULT CRCCore::SetSceneToWindow(int idWindow, int idScene)
{
    if (WindowContainer() == nullptr || SceneContainer() == nullptr) return E_FAIL;
    if (idWindow == CRC::INVALID_ID || idScene == CRC::INVALID_ID) return E_FAIL;
    if (idWindow >= WindowContainer()->GetSize() || idScene >= SceneContainer()->GetSize()) return E_FAIL;

    std::unique_ptr<CRCContainer>& container = WindowContainer();
    std::lock_guard<std::mutex> lock(container->mtx);

    std::unique_ptr<CRCWindowData> windowData = CRC::GetAs<CRCWindowData>(container->Take(idWindow));
    if (!windowData) return E_FAIL;

    windowData->idScene_ = idScene;
}
