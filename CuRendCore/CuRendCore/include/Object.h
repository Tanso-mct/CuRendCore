#pragma once

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <string>

namespace CRC
{

class ObjectController;

typedef struct CRC_API _OBJECT_ATTRIBUTES
{
    std::string name = "";
    std::shared_ptr<ObjectController> ctrl = nullptr;
} OBJECTATTR;

class Object
{
private:
    std::shared_ptr<ObjectController> ctrl = nullptr;

    Object(OBJECTATTR oattr);

    CRC_SLOT thisSlot = CRC_SLOT_INVALID;
    OBJECTATTR oattr;

public:
    ~Object();

    CRC_SLOT GetSlot() { return thisSlot; }

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
};

class CRC_API ObjectFactory
{
private:
    ObjectFactory() { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };
    std::vector<std::shared_ptr<Object>> objects;

public:
    ~ObjectFactory();

    CRC_SLOT CreateObject(OBJECTATTR oattr);
    HRESULT DestroyObject(CRC_SLOT slotObject);

    friend class Group;
};



} // namespace CRC