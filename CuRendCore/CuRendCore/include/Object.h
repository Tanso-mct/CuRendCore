#pragma once

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <string>

#include "Component.h" 

namespace CRC
{

class ObjectController;

typedef struct CRC_API _OBJECT_ATTRIBUTES
{
    std::string name = "";
    std::unique_ptr<ObjectController> ctrl = nullptr;
    CRC_SLOT slotRc = CRC_SLOT_INVALID;
    CRC_SLOT slotBaseTex = CRC_SLOT_INVALID;
    CRC_SLOT slotNormalTex = CRC_SLOT_INVALID;
} OBJECTATTR;

class Object : public Component
{
private:
    Object(OBJECTATTR& oattr);

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

    CRC_SLOT CreateObject(OBJECTATTR& oattr);
    HRESULT DestroyObject(CRC_SLOT slotObject);

    friend class Group;
};



} // namespace CRC