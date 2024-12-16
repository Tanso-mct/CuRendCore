#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <string>

#include "Component.h" 
#include "Math.cuh"

namespace CRC
{

class ObjectController;

typedef struct CRC_API _OBJECT_ATTRIBUTES
{
    std::string name = "";
    std::unique_ptr<ObjectController> ctrl = nullptr;
    CRC_OBJECT_FROM from = CRC_OBJECT_FROM_NULL;
    CRC_SLOT slotRc = CRC_SLOT_INVALID;
    CRC_SLOT slotBaseTex = CRC_SLOT_INVALID;
    CRC_SLOT slotNormalTex = CRC_SLOT_INVALID;

    Vec3d pos = Vec3d(0.0, 0.0, 0.0);
    Vec3d rot = Vec3d(0.0, 0.0, 0.0);
    Vec3d scl = Vec3d(1.0, 1.0, 1.0);
} OBJECT_ATTR;

class Object : public Component
{
private:
    Object(OBJECT_ATTR& oattr);

    // Object attributes.
    std::string name = "";
    std::unique_ptr<ObjectController> ctrl = nullptr;
    CRC_SLOT slotRc = CRC_SLOT_INVALID;
    CRC_SLOT slotBaseTex = CRC_SLOT_INVALID;
    CRC_SLOT slotNormalTex = CRC_SLOT_INVALID;

public:
    ~Object();

    friend class ObjectController;
    friend class ObjectFactory;

    friend class FromObj;
};

class CRC_API ObjectController
{
private:
    std::weak_ptr<Object> object;

    void SetObject(std::weak_ptr<Object> object) { this->object = object; }
    std::shared_ptr<Object> GetObject() { return object.lock(); }

public:
    ObjectController() { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };
    virtual ~ObjectController() { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };

    friend class ObjectFactory;
};

class CRC_API ObjectFactory
{
private:
    ObjectFactory() { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };
    std::vector<std::shared_ptr<Object>> objects;

public:
    ~ObjectFactory();

    CRC_SLOT CreateObject(OBJECT_ATTR& oattr);
    std::shared_ptr<Object> GetObject(CRC_SLOT slotObject);
    HRESULT DestroyObject(CRC_SLOT slotObject);

    friend class Group;
};



} // namespace CRC