#include "Group.h"

namespace CRC
{

Group::Group(GROUPATTR gattr)
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    name = gattr.name;

    objectFc = std::shared_ptr<ObjectFactory>(new ObjectFactory());
    utilityFc = std::shared_ptr<UtilityFactory>(new UtilityFactory());
    uiFc = std::shared_ptr<UIFactory>(new UIFactory());
}

Group::~Group()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    if (objectFc != nullptr) objectFc.reset();
    if (utilityFc != nullptr) utilityFc.reset();
    if (uiFc != nullptr) uiFc.reset();
}

} // namespace CRC