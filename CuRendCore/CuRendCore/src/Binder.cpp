#include "Binder.h"

namespace CRC
{

Binder::Binder(BINDERATTR battr)
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");
    this->battr = battr;
}

Binder::~Binder()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");
}


} // namespace CRC