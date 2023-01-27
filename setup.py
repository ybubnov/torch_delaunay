from pathlib import Path
from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="torchgis",

    package_dir={"torchgis": "torchgis"},
    packages=["torchgis"],

    ext_modules=[
        CppExtension(
            name="torchgis._c",
            sources=[
                "src/cartesian.cc",
                "torchgis/__bind__/cartesian.cc",
            ],
            include_dirs=[
                str(Path.cwd() / "include"),
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
