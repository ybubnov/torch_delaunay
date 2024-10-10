/// Copyright (C) 2024, Yakau Bubnou
///
/// This program is free software: you can redistribute it and/or modify
/// it under the terms of the GNU General Public License as published by
/// the Free Software Foundation, either version 3 of the License, or
/// (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
