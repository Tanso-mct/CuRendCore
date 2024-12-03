#pragma once

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

namespace CRC 
{

class ResourceController;

typedef struct CRC_API _RESOURCE_ATTRIBUTES
{
    std::string path = "";
    std::shared_ptr<ResourceController> ctrl = nullptr;
} RESOURCEATTR;

class CRC_API Resource
{
protected:
    Resource(RESOURCEATTR rattr);

    RESOURCEATTR rattr;
    std::shared_ptr<ResourceController> ctrl = nullptr;

public:
    virtual ~Resource();

    virtual HRESULT Load() = 0;
    virtual HRESULT Unload() = 0;

    friend class ResourceFactory;
    friend class ResourceController;
};

class CRC_API ResourceController
{
protected:
    std::shared_ptr<Resource> resource;

public:
    ResourceController(){CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};
    virtual ~ResourceController(){CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};

    virtual HRESULT OnLoad(HRESULT hrLoad);
    virtual HRESULT OnUnload(HRESULT hrLoad);
    virtual HRESULT Edit() { return S_OK; };

    friend class ResourceFactory;
};

class CRC_API ResourceFactory
{
private:
    ResourceFactory(){CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};
    std::vector<std::shared_ptr<Resource>> resources;
    std::vector<std::string> files;

public:
    ~ResourceFactory();

    CRC_SLOT CreateResource(RESOURCEATTR rattr);
    HRESULT DestroyResource(CRC_SLOT slot);

    HRESULT LoadResource(CRC_SLOT slot);
    HRESULT UnloadResource(CRC_SLOT slot);

    friend class CuRendCore;
};



}