#include "Resource.cuh"
#include "Files.cuh"

namespace CRC 
{

Resource::Resource(RESOURCE_ATTR& rattr)
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    this->path = rattr.path;

    if (rattr.ctrl != nullptr) ctrl = std::move(rattr.ctrl);
    else ctrl = std::unique_ptr<ResourceController>(new ResourceController());
}

Resource::~Resource()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");
    
    if (ctrl != nullptr) ctrl.reset();
}

ResourceFactory::~ResourceFactory()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");
    
    for (auto& resource : resources)
    {
        resource.reset();
    }
    resources.clear();
}

CRC_SLOT ResourceFactory::CreateResource(RESOURCE_ATTR& rattr)
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
    if (slot >= resources.size()) return E_FAIL;
    if (resources[slot] == nullptr) return E_FAIL;

    resources[slot].reset();
    files[slot] = "";
    
    return S_OK;
}

HRESULT ResourceFactory::LoadResource(CRC_SLOT slot)
{
    if (slot >= resources.size()) return E_FAIL;
    if (resources[slot] == nullptr) return E_FAIL;

    HRESULT hr = resources[slot]->Load();
    resources[slot]->ctrl->OnLoad(hr);

    if (FAILED(hr)) return E_FAIL;
    return S_OK;

}

HRESULT ResourceFactory::UnloadResource(CRC_SLOT slot)
{
    if (slot >= resources.size()) return E_FAIL;
    if (resources[slot] == nullptr) return E_FAIL;

    HRESULT hr = resources[slot]->Unload();
    resources[slot]->ctrl->OnUnload(hr);

    if (FAILED(hr)) return E_FAIL;
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