#pragma once

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <unordered_map>

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

    SCENEATTR sattr = { 0 };
    std::shared_ptr<SceneController> ctrl = nullptr;

public:
    ~Scene();

    friend class SceneFactory;
};

class CRC_API SceneController
{
private:
    std::shared_ptr<Scene> scene;

public:
    SceneController() = default;
    virtual ~SceneController() = default;

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