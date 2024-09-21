from pathlib import Path
from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="torchgis",
    package_dir={"torchgis": "torchgis"},
    packages=["torchgis"],
    ext_modules=[
        CppExtension(
            name="torch_delaunay._C",
            sources=[
                "src/triangle.cc",
                "src/predicates.cc",
                "torch_delaunay/__bind__/python_module.cc",
            ],
            include_dirs=[
                str(Path.cwd() / "include"),
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
