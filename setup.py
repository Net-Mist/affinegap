from Cython.Build import cythonize
from setuptools import Extension, setup

ext_modules = cythonize(
    [
        Extension(
            "affinegap.affinegap",
            ["affinegap/affinegap.pyx"],
            language="c++",
            extra_compile_args=["-O3", "-ffast-math", "-march=native"],
        )
    ],
    annotate=True,
)

setup(
    name="affinegap",
    url="https://github.com/datamade/affinegap",
    version="1.12",
    description="A Cython implementation of the affine gap string distance",
    packages=["affinegap"],
    ext_modules=ext_modules,
    license="The MIT License: http://www.opensource.org/licenses/mit-license.php",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: Cython",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
