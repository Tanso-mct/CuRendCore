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
    std::shared_ptr<SceneController> ctrl = nullptr;
} SCENEATTR;

class CRC_API Scene
{
private:
    std::shared_ptr<SceneController> ctrl = nullptr;

    Scene(SCENEATTR sattr); 

    CRC_SLOT thisSlot = CRC_SLOT_INVALID;
    SCENEATTR sattr;
    std::vector<CRC_SLOT> slotRcs;

    std::shared_ptr<GroupFactory> groupFc = nullptr;

public:
    ~Scene();

    CRC_SLOT GetSlot() {return thisSlot;}

    friend class SceneController;
    friend class SceneFactory;
};

class CRC_API SceneController
{
private:
    std::weak_ptr<Scene> scene;
    std::weak_ptr<Input> input;

    bool isWillDestroy = false;
    bool isInited = false;
    bool isReStart = false;

    void SetScene(std::weak_ptr<Scene> scene);
    std::shared_ptr<Scene> GetScene() {return scene.lock();}

    void OnPaint();

protected:
    void AddResource(CRC_SLOT slotResource);
    void RemoveResource(CRC_SLOT slotResource);
    void LoadResources();

    void NeedInit() {isInited = false;}
    void NeedReStart() {isReStart = true;}
    void NeedDestroy(bool val) {isWillDestroy = val;}

    std::shared_ptr<Input> GetInput() {return input.lock();}

    CRC_SLOT CreateGroup(GROUPATTR gattr);
    HRESULT DestroyGroup(CRC_SLOT slotGroup);

public:
    SceneController(){CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};
    virtual ~SceneController(){CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};

    bool IsWillDestroy() {return isWillDestroy;}
    void UnLoadResources();

    CRC_SLOT GetSlot() {return GetScene()->GetSlot();}

    virtual void Init() = 0;
    virtual void Update() = 0;
    virtual void ReStart() = 0;
    virtual void End() = 0;

    bool Finish();

    friend class SceneFactory;
    friend class WindowFactory;
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

    friend class CuRendCore;
};


}