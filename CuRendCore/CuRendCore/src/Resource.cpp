#include "Resource.h"
#include "Files.h"

namespace CRC 
{

Resource::Resource(RESOURCEATTR rattr)
{
    this->rattr = rattr;
    if (rattr.ctrl != nullptr) ctrl = rattr.ctrl;
    else ctrl = std::shared_ptr<ResourceController>(new ResourceController());
}

Resource::~Resource()
{
    if (ctrl != nullptr) ctrl.reset();
}

ResourceFactory::~ResourceFactory()
{
    for (auto& resource : resources)
    {
        resource.reset();
    }
    resources.clear();
}

ResourceFactory *ResourceFactory::GetInstance()
{
    // Implementation of the Singleton pattern.
    static ResourceFactory* instance = nullptr;

    if (instance == nullptr) instance = new ResourceFactory();

    return instance;
}

void ResourceFactory::ReleaseInstance()
{
    ResourceFactory* instance = GetInstance();
    if (instance != nullptr)
    {
        delete instance;
        instance = nullptr;
    }
}

CRC_SLOT ResourceFactory::CreateResource(RESOURCEATTR rattr)
{
    for (int i = 0; i < files.size(); i++)
    {
        if (files[i] == rattr.path) return i;
    }

    std::string extension = rattr.path.substr(rattr.path.find_last_of(".") + 1);
    
    if (extension == "png")
    {
        std::shared_ptr<PngFile> png = std::shared_ptr<PngFile>(new PngFile(rattr));
        png->ctrl->resource = png;
        resources.push_back(png);

        files.push_back(rattr.path);
        return resources.size() - 1;
    }
    else if (extension == "obj")
    {
        std::shared_ptr<ObjFile> obj = std::shared_ptr<ObjFile>(new ObjFile(rattr));
        obj->ctrl->resource = obj;
        resources.push_back(obj);

        files.push_back(rattr.path);
        return resources.size() - 1;
    }
    else
    {
        return CRC_SLOT_INVALID;
    }
    
}

HRESULT ResourceFactory::DestroyResource(CRC_SLOT slot)
{
    if (slot == CRC_SLOT_INVALID) return E_FAIL;

    resources[slot].reset();
    files[slot] = "";
    
    return S_OK;
}

HRESULT ResourceFactory::LoadResource(CRC_SLOT slot)
{
    if (slot == CRC_SLOT_INVALID) return E_FAIL;

    CRC_GRESULT hr = resources[slot]->Load();
    resources[slot]->ctrl->OnLoad(hr);

    if (CRC_GFAILED(hr)) return E_FAIL;
    return S_OK;

}

HRESULT ResourceFactory::UnloadResource(CRC_SLOT slot)
{
    if (slot == CRC_SLOT_INVALID) return E_FAIL;

    CRC_GRESULT hr = resources[slot]->Unload();
    resources[slot]->ctrl->OnUnload(hr);

    if (CRC_GFAILED(hr)) return E_FAIL;
    return S_OK;
}

HRESULT ResourceController::OnLoad(HRESULT hrLoad)
{
    if (FAILED(hrLoad)) return E_FAIL;
    return S_OK;
}

HRESULT ResourceController::OnUnload(HRESULT hrLoad)
{
    if (FAILED(hrLoad)) return E_FAIL;
    return S_OK;
}

} // namespace CRC