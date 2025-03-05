#pragma once

#include "CRC_config.h"

#include <memory>

namespace CRC
{

template <typename T, typename S>
T* As(S* source)
{
    T* target = dynamic_cast<T*>(source);
    return target;
}

template <typename T, typename S>
std::unique_ptr<T> UniqueAs(std::unique_ptr<S>& source)
{
    T* target = dynamic_cast<T*>(source.get());

    if (target) return std::unique_ptr<T>(static_cast<T*>(source.release()));
    else return nullptr;
}

}

template <typename T, typename S>
class CRC_API CRCTransCastUnique
{
private:
    std::unique_ptr<S>& src_;
    std::unique_ptr<T> casted_ = nullptr;

public:
    CRCTransCastUnique(std::unique_ptr<S>& src) : src_(src)
    {
        T* target = dynamic_cast<T*>(src.get());
        if (target) casted_ = std::unique_ptr<T>(static_cast<T*>(src.release()));
    }

    ~CRCTransCastUnique()
    {
        src_ = std::move(casted_);
    }

    std::unique_ptr<T>& operator()() 
    {
        return casted_;
    }
};