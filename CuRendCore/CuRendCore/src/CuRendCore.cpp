#include "CRC_pch.h"

#include "CuRendCore.h"

std::unique_ptr<CRCCore> CRC_API CRC::CreateCRCCore()
{
    return std::make_unique<CRCCore>();
}