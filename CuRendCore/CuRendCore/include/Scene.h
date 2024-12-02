#pragma once

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

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
    Scene(SCENEATTR sattr); 
    CRC_SLOT thisSlot = CRC_SLOT_INVALID;

    SCENEATTR sattr;
    std::shared_ptr<SceneController> ctrl = nullptr;

    std::vector<CRC_SLOT> slotRcs;

public:
    ~Scene();

    CRC_SLOT GetSlot() {return thisSlot;}

    friend class SceneController;
    friend class SceneFactory;
};

class CRC_API SceneController
{
private:
    bool isWillDestroy = false;
    bool isInited = false;
    bool isReStart = false;
    std::shared_ptr<Scene> scene;

    void SetScene(std::shared_ptr<Scene> scene);

protected:
    void AddResource(CRC_SLOT slotResource);
    void RemoveResource(CRC_SLOT slotResource);
    void LoadResources();

    void NeedInit() {isInited = false;}
    void NeedReStart() {isReStart = true;}
    void NeedDestroy(bool val) {isWillDestroy = val;}

public:
    SceneController() = default;
    virtual ~SceneController() = default;

    bool IsWillDestroy() {return isWillDestroy;}
    void UnLoadResources();

    void OnPaint();

    CRC_SLOT GetSlot() {return scene->GetSlot();}

    virtual void Init() = 0;
    virtual void Update() = 0;
    virtual void ReStart() = 0;
    virtual void End() = 0;

    bool Finish();

    friend class SceneFactory;
};

class CRC_API SceneFactory
{
private:
    SceneFactory() = default;
    std::vector<std::shared_ptr<Scene>> scenes;
    
public:
    ~SceneFactory();

    static SceneFactory* GetInstance();
    void ReleaseInstance();

    CRC_SLOT CreateScene(SCENEATTR sattr);
    HRESULT DestroyScene(CRC_SLOT slot);
};


}