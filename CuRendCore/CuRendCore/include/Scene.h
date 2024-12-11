#pragma once

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <string>

#include "Input.h"
#include "Group.h"

namespace CRC 
{

class SceneMani;

typedef struct CRC_API _SCENE_ATTRIBUTES
{
    std::string name = "";
    SceneMani* sceneMani = nullptr;
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

    SceneMani* sceneMani = nullptr;

    bool isStarted = false;
    bool isReStarting = false;
    bool isDestroying = false;

public:
    ~Scene();

    Scene(const Scene&) = delete; // Delete copy constructor
    Scene& operator=(const Scene&) = delete; // Remove copy assignment operator

    Scene(Scene&&) = delete; // Delete move constructor
    Scene& operator=(Scene&&) = delete; // Delete move assignment operator

    SceneMani* GetSceneMani(){return sceneMani;};
    void ManiSetUp(std::weak_ptr<Scene> scene, std::weak_ptr<Input> input);

    CRC_SCENE_STATE Execute();
    void Close();

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
};

class CRC_API SceneMani
{
private:
    std::weak_ptr<Scene> scene;
    std::weak_ptr<Input> input;

protected:
    virtual CRC_SCENE_STATE Start() = 0;
    virtual CRC_SCENE_STATE Update() = 0;
    virtual CRC_SCENE_STATE Destroy() = 0;

    virtual CRC_SCENE_STATE ReStart() = 0;

    // Obtain a scene.It is not recommended to use this by storing it in a non-temporary variable.
    std::shared_ptr<Scene> GetScene(){return scene.lock();};

    // Get the scene's weak_ptr unlike GetScene, there is no problem storing it in a non-temporary variable for use.
    std::weak_ptr<Scene> GetSceneWeak(){return scene;};

    // Obtain an input.It is not recommended to use this by storing it in a non-temporary variable.
    std::shared_ptr<Input> GetInput(){return input.lock();};

    // Get the input's weak_ptr unlike GetInput, there is no problem storing it in a non-temporary variable for use.
    std::weak_ptr<Input> GetInputWeak(){return input;};

public:
    SceneMani() {CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};
    virtual ~SceneMani() {CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};

    friend class Scene;
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