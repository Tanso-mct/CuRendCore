#pragma once

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <string>

#include "Binder.h"
#include "Object.h"
#include "Utility.h"
#include "UI.h"

namespace CRC 
{

class BinderController;

typedef struct CRC_API _BINDER_ATTRIBUTES
{
    std::string name = "";
    std::shared_ptr<BinderController> ctrl = nullptr;
} BINDERATTR;

class CRC_API Binder
{
private:
    std::shared_ptr<BinderController> ctrl = nullptr;

    Binder(BINDERATTR battr); 

    CRC_SLOT thisSlot = CRC_SLOT_INVALID;
    BINDERATTR battr;

    std::vector<std::weak_ptr<Group>> groups;
    std::vector<std::weak_ptr<Object>> objects;
    std::vector<std::weak_ptr<Utility>> utilities;
    std::vector<std::weak_ptr<UI>> uis;

public:
    virtual ~Binder();

    CRC_SLOT GetSlot() {return thisSlot;}

    friend class GroupFactory;
};

} // namespace CRC