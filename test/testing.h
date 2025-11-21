// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

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
