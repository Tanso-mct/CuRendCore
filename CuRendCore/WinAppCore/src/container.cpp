#include "pch.h"
#include "container.h"

int WACore::Container::Add(std::unique_ptr<WACore::IContainable> data)
{
    datas_.emplace_back(std::move(data));
    return datas_.size() - 1;
}

HRESULT WACore::Container::Del(int id)
{
    if (id < 0 || id >= datas_.size()) return E_FAIL;

    datas_[id].reset();
    return S_OK;
}

std::unique_ptr<WACore::IContainable> &WACore::Container::Get(int id)
{
    if (id < 0 || id >= datas_.size()) return emptyData_;
    return datas_[id];
}

std::unique_ptr<WACore::IContainable> WACore::Container::Take(int id)
{
    if (id < 0 || id >= datas_.size()) return nullptr;
    return std::move(datas_[id]);
}

HRESULT WACore::Container::Put(int id, std::unique_ptr<WACore::IContainable> data)
{
    if (id < 0 || id >= datas_.size()) return E_FAIL;
    
    datas_[id] = std::move(data);
    return S_OK;
}

int WACore::Container::GetSize()
{
    return datas_.size();
}

void WACore::Container::Clear()
{
    datas_.clear();
}
