#pragma once

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <string>

#include "Object.cuh"
#include "Utility.cuh"
#include "UI.cuh"

#include "Binder.h"

namespace CRC 
{

typedef struct CRC_API _GROUP_ATTRIBUTES
{
    std::string name = "";
    BOOL active = TRUE;
} GROUPATTR;

class CRC_API Group
{
private:
    Group(GROUPATTR gattr); 

    CRC_SLOT thisSlot = CRC_SLOT_INVALID;
    
    // Group attributes.
    std::string name = "";
    BOOL active = TRUE;

    std::shared_ptr<ObjectFactory> objectFc = nullptr;
    std::shared_ptr<UtilityFactory> utilityFc = nullptr;
    std::shared_ptr<UIFactory> uiFc = nullptr;

public:
    ~Group();

    Group(const Group&) = delete; // Delete copy constructor
    Group& operator=(const Group&) = delete; // Remove copy assignment operator

    Group(Group&&) = delete; // Delete move constructor
    Group& operator=(Group&&) = delete; // Delete move assignment operator

    CRC_SLOT GetSlot() {return thisSlot;}

    friend class Scene;
};

} // namespace CRC