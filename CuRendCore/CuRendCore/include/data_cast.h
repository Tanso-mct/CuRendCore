#pragma once

#include "CuRendCore/include/config.h"

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

template <typename D, typename S>
class CRC_API CRCTransCastUnique
{
private:
    std::unique_ptr<S>& src_;
    std::unique_ptr<D> casted_ = nullptr;

public:
    CRCTransCastUnique(std::unique_ptr<S>& src) : src_(src)
    {
        D* dstPtr = dynamic_cast<D*>(src.get());
        if (dstPtr) casted_.reset(dynamic_cast<D*>(src.release()));
    }

    ~CRCTransCastUnique()
    {
        src_.reset(dynamic_cast<S*>(casted_.release()));
    }

    std::unique_ptr<D>& operator()() 
    {
        return casted_;
    }
};