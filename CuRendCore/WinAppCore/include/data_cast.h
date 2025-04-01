#pragma once
#include "config.h"

#include <memory>

namespace WACore
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

/**
 * Casts an instance of type unique_ptr to the specified type and 
 * returns ownership to the cast source when it exits scope.
 * Casting from one type of RevertCast to another is discouraged.
 */
template <typename D, typename S>
class WIN_APP_CORE RevertCast
{
private:
    std::unique_ptr<S>& src_;
    std::unique_ptr<D> casted_ = nullptr;

public:
    RevertCast(std::unique_ptr<S>& src) : src_(src)
    {
        D* dstPtr = dynamic_cast<D*>(src.get());
        if (dstPtr) casted_.reset(dynamic_cast<D*>(src.release()));
    }

    ~RevertCast()
    {
        src_.reset(dynamic_cast<S*>(casted_.release()));
    }

    std::unique_ptr<D>& operator()() 
    {
        return casted_;
    }
};

} // namespace WACore
