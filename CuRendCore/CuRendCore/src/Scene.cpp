#include "Scene.h"
#include "Resource.h"

namespace CRC 
{

Scene::Scene(SCENEATTR sattr)
{
    this->sattr = sattr;
    this->ctrl = sattr.ctrl;
}

Scene::~Scene()
{
    if (ctrl != nullptr) ctrl.reset();

    ResourceFactory* rf = ResourceFactory::GetInstance();
    for (auto& slot : slotRcs)
    {
        rf->UnloadResource(slot);
    }
}

SceneFactory::~SceneFactory()
{
    for (auto& scene : scenes)
    {
        scene.reset();
    }
    scenes.clear();
}

SceneFactory *SceneFactory::GetInstance()
{
    // Implementation of the Singleton pattern.
    static SceneFactory* instance = nullptr;

    if (instance == nullptr) instance = new SceneFactory();

    return instance;
}

void SceneFactory::ReleaseInstance()
{
    SceneFactory* instance = GetInstance();
    if (instance != nullptr)
    {
        delete instance;
        instance = nullptr;
    }
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

    scenes[slot].reset();
    return S_OK;
}

void SceneController::SetScene(std::shared_ptr<Scene> scene)
{
    this->scene = scene;
    isInited = false;
}

void SceneController::AddResource(CRC_SLOT slotResource)
{
    for (auto& scene : scene->slotRcs)
    {
        if (scene == slotResource) return;
    }

    scene->slotRcs.push_back(slotResource);
}

void SceneController::RemoveResource(CRC_SLOT slotResource)
{
    int removeId = -1;
    for (int i = 0; i < scene->slotRcs.size(); i++)
    {
        if (scene->slotRcs[i] == slotResource)
        {
            removeId = i;
            break;
        }
    }

    if (removeId != -1) scene->slotRcs.erase(scene->slotRcs.begin() + removeId);
}

void SceneController::LoadResources()
{
    ResourceFactory* rf = ResourceFactory::GetInstance();
    for (int i = 0; i < scene->slotRcs.size(); i++)
    {
        rf->LoadResource(scene->slotRcs[i]);
    }
}

void SceneController::UnLoadResources()
{
    ResourceFactory* rf = ResourceFactory::GetInstance();
    for (int i = 0; i < scene->slotRcs.size(); i++)
    {
        rf->UnloadResource(scene->slotRcs[i]);
    }
}

void SceneController::OnPaint()
{
    if (isReStart)
    {
        scene->ctrl->ReStart();
        isReStart = false;
        return;
    }

    if (!isInited)
    {
        scene->ctrl->Init();
        isInited = true;
        return;
    }

    scene->ctrl->Update();
}

bool SceneController::Finish()
{
    End();
    if (IsWillDestroy())
    {
        // The scene will be destroyed.
        SceneFactory::GetInstance()->DestroyScene(GetSlot());
        return true;
    }

    return false;
}

 } // namespace CRC
