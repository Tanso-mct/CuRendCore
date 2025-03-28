#include "pch.h"

#include <string_view>

int main(int argc, char **argv)
{
    #ifdef __cpp_lib_string_view
    std::cout << "std::string_view is available\n";
#else
    std::cout << "std::string_view is NOT available\n";
#endif

    std::string_view sv("Hello, World!");

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}