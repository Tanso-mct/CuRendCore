#pragma once

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <string>

#include "Binder.h"
#include "Object.cuh"
#include "Utility.cuh"
#include "UI.cuh"

namespace CRC 
{

class CRC_API Binder
{
private:
    CRC_SLOT thisSlot = CRC_SLOT_INVALID;

    std::vector<std::weak_ptr<Group>> groups;
    std::vector<std::weak_ptr<Object>> objects;
    std::vector<std::weak_ptr<Utility>> utilities;
    std::vector<std::weak_ptr<UI>> uis;

public:
    Binder(){CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};
    virtual ~Binder(){CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};

    Binder(Binder&&) = delete; // Delete move constructor
    Binder& operator=(Binder&&) = delete; // Delete move assignment operator

    CRC_SLOT GetSlot() {return thisSlot;}

    friend class Scene;
};

} // namespace CRC