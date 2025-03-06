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
        if (dstPtr) casted_ = std::unique_ptr<D>(static_cast<D*>(src.release()));
    }

    ~CRCTransCastUnique()
    {
        src_ = std::move(casted_);
    }

    std::unique_ptr<D>& operator()() 
    {
        return casted_;
    }
};

template <typename PARENT, typename D, typename S>
class CRC_API CRCTransCastUniqueItoI
{
private:
    std::unique_ptr<S>& src_;
    std::unique_ptr<D> casted_ = nullptr;

public:
    CRCTransCastUniqueItoI(std::unique_ptr<S>& src) : src_(src)
    {
        PARENT* parentPtr = dynamic_cast<PARENT*>(src.get());
        if (parentPtr)
        {
            std::unique_ptr<PARENT> parent  = std::unique_ptr<PARENT>(static_cast<PARENT*>(src.release()));
            casted_ = std::move(parent);
        }
    }

    ~CRCTransCastUniqueItoI()
    {
        PARENT* parentPtr = dynamic_cast<PARENT*>(casted_.get());
        if (parentPtr)
        {
            std::unique_ptr<PARENT> parent  = std::unique_ptr<PARENT>(static_cast<PARENT*>(casted_.release()));
            src_ = std::move(parent);
        }
    }

    std::unique_ptr<D>& operator()() 
    {
        return casted_;
    }
};