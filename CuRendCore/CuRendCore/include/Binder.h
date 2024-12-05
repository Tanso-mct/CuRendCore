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

typedef struct CRC_API _BINDER_ATTRIBUTES
{
    std::string name = "";
} BINDERATTR;

class CRC_API Binder
{
private:
    CRC_SLOT thisSlot = CRC_SLOT_INVALID;
    BINDERATTR battr;

    std::vector<std::weak_ptr<Group>> groups;
    std::vector<std::weak_ptr<Object>> objects;
    std::vector<std::weak_ptr<Utility>> utilities;
    std::vector<std::weak_ptr<UI>> uis;

public:
    Binder(BINDERATTR battr);
    virtual ~Binder();

    CRC_SLOT GetSlot() {return thisSlot;}

    friend class SceneController;
};

} // namespace CRC