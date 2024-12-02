#include "Scene.h"

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
}

SceneFactory::~SceneFactory()
{
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
    newScene->ctrl->scene = newScene;
    scenes.push_back(newScene);

    return (CRC_SLOT)(scenes.size() - 1);
}

HRESULT SceneFactory::DestroyScene(CRC_SLOT slot)
{
    if (slot >= scenes.size()) return E_FAIL;

    scenes[slot].reset();
    scenes.erase(scenes.begin() + slot);
    return S_OK;
}

HRESULT SceneController::SetName(std::string name)
{
    scene->sattr.name = name;
    return S_OK;
}

} // namespace CRC
