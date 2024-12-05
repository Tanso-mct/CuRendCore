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

typedef struct CRC_API _GROUP_ATTRIBUTES
{
    std::string name = "";
} GROUPATTR;

class CRC_API Group
{
private:
    Group(GROUPATTR gattr); 

    CRC_SLOT thisSlot = CRC_SLOT_INVALID;
    GROUPATTR gattr;

    std::shared_ptr<ObjectFactory> objectFc = nullptr;
    std::shared_ptr<UtilityFactory> utilityFc = nullptr;
    std::shared_ptr<UIFactory> uiFc = nullptr;

public:
    ~Group();

    CRC_SLOT GetSlot() {return thisSlot;}

    friend class SceneController;
};

} // namespace CRC