# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Yakau Bubnou
# SPDX-FileType: SOURCE

from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    ext_modules=[
        CppExtension(
            name="torch_delaunay._C",
            sources=[
                "src/predicate.cc",
                "src/sweephull.cc",
                "src/triangle.cc",
                "torch_delaunay/__bind__/python_module.cc",
            ],
            include_dirs=[
                str(Path.cwd() / "include"),
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
