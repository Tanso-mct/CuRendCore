#include "CuRendCore/include/pch.h"
#include "CuRendCore/include/funcs.cuh"

#include "CuRendCore/include/window.h"

std::unique_ptr<ICRCContainable> CRCWindowFactory::Create(IDESC &desc) const
{
    CRC_WINDOW_DESC* windowDesc = WACore::As<CRC_WINDOW_DESC>(&desc);
    if (!&windowDesc) return nullptr;

    std::unique_ptr<CRCWindowAttr> windowAttr = std::make_unique<CRCWindowAttr>();

    if (!RegisterClassEx(&windowDesc->wcex_)) return nullptr;

    windowAttr->hWnd_ = CreateWindow
    (
        windowDesc->wcex_.lpszClassName,
        windowDesc->name_,
        windowDesc->style_,
        windowDesc->initialPosX_,
        windowDesc->initialPosY_,
        windowDesc->width_,
        windowDesc->height_,
        windowDesc->hWndParent_,
        nullptr,
        windowDesc->hInstance,
        nullptr
    );
    if (!windowAttr->hWnd_) return nullptr;

    return windowAttr;
}
