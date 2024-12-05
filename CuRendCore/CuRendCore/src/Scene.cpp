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

    for (auto& binder : binders)
    {
        binder.reset();
    }

    for (auto& group : groups)
    {
        group.reset();
    }

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

CRC_SLOT SceneController::CreateGroup(GROUPATTR gattr)
{
    std::shared_ptr<Group> newGroup = std::shared_ptr<Group>(new Group(gattr));
    GetScene()->groups.push_back(newGroup);

    newGroup->thisSlot = (CRC_SLOT)(GetScene()->groups.size() - 1);
    return (CRC_SLOT)(GetScene()->groups.size() - 1);
}

HRESULT SceneController::DestroyGroup(CRC_SLOT slotGroup)
{
    if (slotGroup >= GetScene()->groups.size()) return E_FAIL;
    if (GetScene()->groups[slotGroup] == nullptr) return E_FAIL;

    GetScene()->groups[slotGroup].reset();
    return S_OK;
}

CRC_SLOT SceneController::CreateComponent(CRC_SLOT slotGroup, OBJECTATTR oattr)
{
    if (slotGroup >= GetScene()->groups.size()) return CRC_SLOT_INVALID;
    if (GetScene()->groups[slotGroup] == nullptr) return CRC_SLOT_INVALID;

    return GetScene()->groups[slotGroup]->objectFc->CreateObject(oattr);
}

CRC_SLOT SceneController::CreateComponent(CRC_SLOT slotGroup, UTILITYATTR utattr)
{
    if (slotGroup >= GetScene()->groups.size()) return CRC_SLOT_INVALID;
    if (GetScene()->groups[slotGroup] == nullptr) return CRC_SLOT_INVALID;

    return GetScene()->groups[slotGroup]->utilityFc->CreateUtility(utattr);
}

CRC_SLOT SceneController::CreateComponent(CRC_SLOT slotGroup, UIATTR uiattr)
{
    if (slotGroup >= GetScene()->groups.size()) return CRC_SLOT_INVALID;
    if (GetScene()->groups[slotGroup] == nullptr) return CRC_SLOT_INVALID;

    return GetScene()->groups[slotGroup]->uiFc->CreateUI(uiattr);
}

HRESULT SceneController::DestroyComponent(CRC_COMPONENT_TYPE type, CRC_SLOT slotGroup, CRC_SLOT slotComponent)
{
    if (slotGroup >= GetScene()->groups.size()) return E_FAIL;
    if (GetScene()->groups[slotGroup] == nullptr) return E_FAIL;

    switch (type)
    {
    case CRC_COMPONENT_TYPE_OBJECT:
        return GetScene()->groups[slotGroup]->objectFc->DestroyObject(slotComponent);
    case CRC_COMPONENT_TYPE_UTILITY:
        return GetScene()->groups[slotGroup]->utilityFc->DestroyUtility(slotComponent);
    case CRC_COMPONENT_TYPE_UI:
        return GetScene()->groups[slotGroup]->uiFc->DestroyUI(slotComponent);
    default:
        return E_FAIL;
    }
}

CRC_SLOT SceneController::AddBinder(std::unique_ptr<Binder> binder)
{
    binder->thisSlot = (CRC_SLOT)(GetScene()->binders.size());
    GetScene()->binders.push_back(std::move(binder));
    
    return (CRC_SLOT)(GetScene()->binders.size() - 1);
}

HRESULT SceneController::DestroyBinder(CRC_SLOT slotBinder)
{
    if (slotBinder >= GetScene()->binders.size()) return E_FAIL;
    if (GetScene()->binders[slotBinder] == nullptr) return E_FAIL;

    GetScene()->binders[slotBinder].reset();
    return S_OK;
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
