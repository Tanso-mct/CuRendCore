#include "Scene.h"
#include "Resource.cuh"
#include "CuRendCore.h"


namespace CRC 
{

Scene::Scene(SCENEATTR sattr)
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    this->sattr = sattr;
    this->ctrl = sattr.ctrl;
}

Scene::~Scene()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    if (ctrl != nullptr) ctrl.reset();

    ResourceFactory* rf = CuRendCore::GetInstance()->resourceFc;
    for (auto& slot : slotRcs)
    {
        rf->UnloadResource(slot);
    }
}

SceneFactory::~SceneFactory()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    for (auto& scene : scenes)
    {
        scene.reset();
    }
    scenes.clear();
}

CRC_SLOT SceneFactory::CreateScene(SCENEATTR sattr)
{
    std::shared_ptr<Scene> newScene = std::shared_ptr<Scene>(new Scene(sattr));
    newScene->ctrl->SetScene(newScene);
    scenes.push_back(newScene);

    newScene->thisSlot = (CRC_SLOT)(scenes.size() - 1);
    return (CRC_SLOT)(scenes.size() - 1);
}

HRESULT SceneFactory::DestroyScene(CRC_SLOT slot)
{
    if (slot >= scenes.size()) return E_FAIL;
    if (scenes[slot] == nullptr) return E_FAIL;

    scenes[slot].reset();
    return S_OK;
}

void SceneController::SetScene(std::weak_ptr<Scene> scene)
{
    this->scene = scene;
    isInited = false;
}

void SceneController::AddResource(CRC_SLOT slotResource)
{
    for (auto& scene : GetScene()->slotRcs)
    {
        if (scene == slotResource) return;
    }

    GetScene()->slotRcs.push_back(slotResource);
}

void SceneController::RemoveResource(CRC_SLOT slotResource)
{
    int removeId = -1;
    for (int i = 0; i < GetScene()->slotRcs.size(); i++)
    {
        if (GetScene()->slotRcs[i] == slotResource)
        {
            removeId = i;
            break;
        }
    }

    if (removeId != -1) GetScene()->slotRcs.erase(GetScene()->slotRcs.begin() + removeId);
}

void SceneController::LoadResources()
{
    ResourceFactory* rf = CuRendCore::GetInstance()->resourceFc;
    for (int i = 0; i < GetScene()->slotRcs.size(); i++)
    {
        rf->LoadResource(GetScene()->slotRcs[i]);
    }
}

void SceneController::UnLoadResources()
{
    ResourceFactory* rf = CuRendCore::GetInstance()->resourceFc;
    for (int i = 0; i < GetScene()->slotRcs.size(); i++)
    {
        rf->UnloadResource(GetScene()->slotRcs[i]);
    }
}

void SceneController::OnPaint()
{
    if (isReStart)
    {
        GetScene()->ctrl->ReStart();
        isReStart = false;
        return;
    }

    if (!isInited)
    {
        GetScene()->ctrl->Init();
        isInited = true;
        return;
    }

    GetScene()->ctrl->Update();
}

bool SceneController::Finish()
{
    End();
    if (IsWillDestroy())
    {
        // The scene will be destroyed.
        SceneFactory* sf = CuRendCore::GetInstance()->sceneFc;
        sf->DestroyScene(GetSlot());
        return true;
    }

    return false;
}

 } // namespace CRC
