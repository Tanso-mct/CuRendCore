#pragma once

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <string>

#include "Input.h"
#include "Group.h"

namespace CRC 
{

class SceneController;

typedef struct CRC_API _SCENE_ATTRIBUTES
{
    std::string name = "";
} SCENEATTR;

class CRC_API Scene
{
private:
    Scene(SCENEATTR sattr); 

    CRC_SLOT thisSlot = CRC_SLOT_INVALID;
    std::string name = "";

    std::vector<CRC_SLOT> slotResources;

    std::vector<std::unique_ptr<Group>> groups;
    std::vector<std::unique_ptr<Binder>> binders;

    bool isInited = false;
    bool isReStarting = false;
    bool isDestroying = false;

public:
    ~Scene();

    Scene(const Scene&) = delete; // Delete copy constructor
    Scene& operator=(const Scene&) = delete; // Remove copy assignment operator

    Scene(Scene&&) = delete; // Delete move constructor
    Scene& operator=(Scene&&) = delete; // Delete move assignment operator

    // Add and remove resources.
    HRESULT AddResource(CRC_SLOT slotResource);
    HRESULT RemoveResource(CRC_SLOT slotResource);

    // Load added resources.
    HRESULT LoadResources();

    // Unload added resources.
    HRESULT UnLoadResources();

    CRC_SLOT CreateGroup(GROUPATTR gattr);
    HRESULT DestroyGroup(CRC_SLOT slotGroup);

    CRC_SLOT AddBinder(Binder*& binder);
    HRESULT DestroyBinder(CRC_SLOT slotBinder);

    CRC_SLOT CreateComponent(CRC_SLOT slotGroup, OBJECTATTR oattr);
    CRC_SLOT CreateComponent(CRC_SLOT slotGroup, UTILITYATTR utattr);
    CRC_SLOT CreateComponent(CRC_SLOT slotGroup, UIATTR uiattr);

    HRESULT DestroyComponent(CRC_COMPONENT_TYPE type, CRC_SLOT slotGroup, CRC_SLOT slotComponent);

    friend class SceneFactory;
    friend class WindowController;
};

class CRC_API SceneFactory
{
private:
    SceneFactory(){CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};
    std::vector<std::shared_ptr<Scene>> scenes;
    
public:
    ~SceneFactory();

    CRC_SLOT CreateScene(SCENEATTR sattr);
    HRESULT DestroyScene(CRC_SLOT slot);

    std::weak_ptr<Scene> GetSceneWeak(CRC_SLOT slot);

    friend class CuRendCore;
};


} // namespace CRC