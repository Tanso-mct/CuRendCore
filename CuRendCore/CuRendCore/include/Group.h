#pragma once

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <string>

#include "Object.h"
#include "Utility.h"
#include "UI.h"

#include "Binder.h"

namespace CRC 
{

class GroupController;

typedef struct CRC_API _GROUP_ATTRIBUTES
{
    std::string name = "";
    std::shared_ptr<GroupController> ctrl = nullptr;
} GROUPATTR;

class CRC_API Group
{
private:
    std::shared_ptr<GroupController> ctrl = nullptr;

    Group(GROUPATTR gattr); 

    CRC_SLOT thisSlot = CRC_SLOT_INVALID;
    GROUPATTR gattr;

    std::shared_ptr<ObjectFactory> objectFc = nullptr;
    std::shared_ptr<UtilityFactory> utilityFc = nullptr;
    std::shared_ptr<UIFactory> uiFc = nullptr;

public:
    ~Group();

    CRC_SLOT GetSlot() {return thisSlot;}

    friend class GroupController;
    friend class GroupFactory;
};

class CRC_API GroupController
{
private:
    std::weak_ptr<Group> group;

    void SetGroup(std::weak_ptr<Group> group){this->group = group;}
    std::shared_ptr<Group> GetGroup() {return group.lock();}

public:
    GroupController(){CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};
    virtual ~GroupController(){CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};

    friend class GroupFactory;
};

class CRC_API GroupFactory
{
private:
    GroupFactory(){CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};

    std::vector<std::shared_ptr<Group>> groups;
    std::vector<std::shared_ptr<Binder>> binders;

public:
    ~GroupFactory();

    CRC_SLOT CreateGroup(GROUPATTR gattr);
    HRESULT DestroyGroup(CRC_SLOT slotGroup);

    CRC_SLOT CreateBinder(BINDERATTR battr);
    HRESULT DestroyBinder(CRC_SLOT slotBinder);

    friend class Scene;
};


} // namespace CRC