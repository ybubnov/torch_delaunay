#pragma once

#include <functional>


template <typename Exception>
std::function<bool(const Exception&)>
exception_contains_text(const std::string error_message)
{
    return [&](const Exception& error) -> bool {
        return std::string(error.what()).find(error_message) != std::string::npos;
    };
}
