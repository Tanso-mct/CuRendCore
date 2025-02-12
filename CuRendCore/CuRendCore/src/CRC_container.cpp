#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_container.h"

int CRCContainer::Add(std::unique_ptr<ICRCContainable> data)
{
    datas_.emplace_back(std::move(data));
    return datas_.size() - 1;
}

HRESULT CRCContainer::Remove(int id)
{
    if (id < 0 || id >= datas_.size()) return E_FAIL;

    datas_[id].reset();
    return S_OK;
}

std::unique_ptr<ICRCContainable> &CRCContainer::Get(int id)
{
    if (id < 0 || id >= datas_.size()) return emptyData_;
    return datas_[id];
}

std::unique_ptr<ICRCContainable> CRCContainer::Take(int id)
{
    if (id < 0 || id >= datas_.size()) return nullptr;
    return std::move(datas_[id]);
}

HRESULT CRCContainer::Put(int id, std::unique_ptr<ICRCContainable> data)
{
    if (id < 0 || id >= datas_.size()) return E_FAIL;
    
    datas_[id] = std::move(data);
    return S_OK;
}

int CRCContainer::GetSize()
{
    return datas_.size();
}

void CRCContainer::Clear()
{
    datas_.clear();
}

int CRCContainerSet::Add(std::unique_ptr<ICRCContainer> data)
{
    containers_.emplace_back(std::move(data));
    return containers_.size() - 1;
}

HRESULT CRCContainerSet::Remove(int id)
{
    if (id < 0 || id >= containers_.size()) return E_FAIL;
    
    containers_[id].reset();
    return S_OK;
}

std::unique_ptr<ICRCContainer> &CRCContainerSet::Get(int id)
{
    if (id < 0 || id >= containers_.size()) return emptyContainer_;
    return containers_[id];
}

std::unique_ptr<ICRCContainer> CRCContainerSet::Take(int id)
{
    if (id < 0 || id >= containers_.size()) return nullptr;
    return std::move(containers_[id]);
}

HRESULT CRCContainerSet::Put(int id, std::unique_ptr<ICRCContainer> data)
{
    if (id < 0 || id >= containers_.size()) return E_FAIL;
    
    containers_[id] = std::move(data);
    return S_OK;
}

int CRCContainerSet::GetSize()
{
    return containers_.size();
}

void CRCContainerSet::Clear()
{
    containers_.clear();
}
