#include "Scene.h"
#include "Resource.cuh"
#include "CuRendCore.h"


namespace CRC 
{

Scene::Scene(SCENE_ATTR sattr)
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    name = sattr.name;
    sceneMani = sattr.sceneMani;

    if (sceneMani == nullptr)
    {
        CRCErrorMsgBox
        (
            __FILE__, __FUNCTION__, __LINE__, 
            "SATTR's SceneMani is nullptr.\nWhen creating a scene, set SceneMani to the SCENE_ATTR structure."
        );
    }
}

Scene::~Scene()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    for (auto& binder : binders)
    {
        binder.reset();
    }
    binders.clear();

    for (auto& group : groups)
    {
        group.reset();
    }
    groups.clear();

    ResourceFactory* rf = CuRendCore::GetInstance()->resourceFc;
    for (auto& slot : slotResources)
    {
        rf->UnloadResource(slot);
    }
    slotResources.clear();

    if (sceneMani != nullptr)
    {
        delete sceneMani;
        sceneMani = nullptr;
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

CRC_SLOT SceneFactory::CreateScene(SCENE_ATTR sattr)
{
    std::shared_ptr<Scene> newScene = std::shared_ptr<Scene>(new Scene(sattr));
    newScene->thisSlot = (CRC_SLOT)(scenes.size());
    scenes.push_back(newScene);

    return scenes.size() - 1;
}

HRESULT SceneFactory::DestroyScene(CRC_SLOT slot)
{
    if (slot >= scenes.size()) return E_FAIL;
    if (scenes[slot] == nullptr) return E_FAIL;

    scenes[slot].reset();
    return S_OK;
}

std::weak_ptr<Scene> SceneFactory::GetSceneWeak(CRC_SLOT slot)
{
    if (slot >= scenes.size()) return std::weak_ptr<Scene>();
    if (scenes[slot] == nullptr) return std::weak_ptr<Scene>();

    return scenes[slot];
}

void Scene::ManiSetUp(std::weak_ptr<Scene> scene, std::weak_ptr<Input> input)
{
    sceneMani->scene = scene;
    sceneMani->input = input;
}

CRC_SCENE_STATE Scene::Execute()
{
    if (!isStarted && !isReStarting && !isDestroying)
    {
        CRC_SCENE_STATE state = sceneMani->Start();
        isStarted = true;
        return state;
    }
    else if (isStarted && !isReStarting && !isDestroying)
    {
        CRC_SCENE_STATE state = sceneMani->Update();
        return state;
    }

    if (isReStarting && !isDestroying)
    {
        CRC_SCENE_STATE state = sceneMani->ReStart();
        isReStarting = false;
        return state;
    }

    return CRC_SCENE_STATE_ERROR;
}

CRC_SCENE_STATE Scene::Close()
{
    CRC_SCENE_STATE state = sceneMani->End();
    return state;
}

HRESULT Scene::AddResource(CRC_SLOT slotResource)
{
    for (auto& slot : slotResources)
    {
        if (slot == slotResource) return E_FAIL;
    }

    slotResources.push_back(slotResource);
    return S_OK;
}

HRESULT Scene::RemoveResource(CRC_SLOT slotResource)
{
    int removeId = -1;
    for (int i = 0; i < slotResources.size(); i++)
    {
        if (slotResources[i] == slotResource)
        {
            removeId = i;
            break;
        }
    }

    if (removeId != -1) slotResources.erase(slotResources.begin() + removeId);

    return S_OK;
}

HRESULT Scene::LoadResources()
{
    ResourceFactory* rf = CuRendCore::GetInstance()->resourceFc;
    for (int i = 0; i < slotResources.size(); i++)
    {
        if (!SUCCEEDED(rf->LoadResource(slotResources[i]))) return E_FAIL;
    }

    return S_OK;
}

HRESULT Scene::UnLoadResources()
{
    ResourceFactory* rf = CuRendCore::GetInstance()->resourceFc;
    for (int i = 0; i < slotResources.size(); i++)
    {
        if (!SUCCEEDED(rf->UnloadResource(slotResources[i]))) return E_FAIL;
    }

    return S_OK;
}

CRC_SLOT Scene::CreateGroup(GROUPATTR gattr)
{
    Group* newGroup = new Group(gattr);
    newGroup->thisSlot = (CRC_SLOT)(groups.size());
    groups.push_back(std::unique_ptr<Group>(newGroup));

    return groups.size() - 1;
}

HRESULT Scene::DestroyGroup(CRC_SLOT slotGroup)
{
    if (slotGroup >= groups.size()) return E_FAIL;
    if (groups[slotGroup] == nullptr) return E_FAIL;

    groups[slotGroup].reset();
    return S_OK;
}

CRC_SLOT Scene::AddBinder(Binder*& binder)
{
    binder->thisSlot = (CRC_SLOT)(binders.size());
    binders.push_back(std::unique_ptr<Binder>(binder));

    return binders.size() - 1;
}

HRESULT Scene::DestroyBinder(CRC_SLOT slotBinder)
{
    if (slotBinder >= binders.size()) return E_FAIL;
    if (binders[slotBinder] == nullptr) return E_FAIL;

    binders[slotBinder].reset();
    return S_OK;
}

CRC_SLOT Scene::CreateComponent(CRC_SLOT slotGroup, OBJECT_ATTR& oattr)
{
    if (slotGroup >= groups.size()) return CRC_SLOT_INVALID;
    if (groups[slotGroup] == nullptr) return CRC_SLOT_INVALID;

    return groups[slotGroup]->objectFc->CreateObject(oattr);
}

CRC_SLOT Scene::CreateComponent(CRC_SLOT slotGroup, UTILITYATTR& utattr)
{
    if (slotGroup >= groups.size()) return CRC_SLOT_INVALID;
    if (groups[slotGroup] == nullptr) return CRC_SLOT_INVALID;

    return groups[slotGroup]->utilityFc->CreateUtility(utattr);
}

CRC_SLOT Scene::CreateComponent(CRC_SLOT slotGroup, UIATTR& uiattr)
{
    if (slotGroup >= groups.size()) return CRC_SLOT_INVALID;
    if (groups[slotGroup] == nullptr) return CRC_SLOT_INVALID;

    return groups[slotGroup]->uiFc->CreateUI(uiattr);
}

HRESULT Scene::DestroyComponent(CRC_COMPONENT_TYPE type, CRC_SLOT slotGroup, CRC_SLOT slotComponent)
{
    if (slotGroup >= groups.size()) return E_FAIL;
    if (groups[slotGroup] == nullptr) return E_FAIL;

    switch (type)
    {
    case CRC_COMPONENT_TYPE_OBJECT:
        return groups[slotGroup]->objectFc->DestroyObject(slotComponent);
    case CRC_COMPONENT_TYPE_UTILITY:
        return groups[slotGroup]->utilityFc->DestroyUtility(slotComponent);
    case CRC_COMPONENT_TYPE_UI:
        return groups[slotGroup]->uiFc->DestroyUI(slotComponent);
    default:
        return E_FAIL;
    }
}

} // namespace CRC
