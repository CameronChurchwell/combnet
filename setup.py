from setuptools import find_packages, setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import torch
import os
import glob

library_name = "combnet"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False

def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    use_cuda = "0"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() #and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    # extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    # cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    # if use_cuda:
    #     sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules


# modules = [
#     CUDAExtension(
#         'cuda_ops',
#         [
#             'combnet/cuda_ops.cpp',
#             'combnet/sparse_conv1d.cu'
#         ],
#         # extra_compile_args={'cxx': [], 'nvcc': ['-keep', '-G', '-O3', '--source-in-ptx']}
#         extra_compile_args={'cxx': ['-fopenmp', '-O3'], 'nvcc': ['-O3']}
#     )
# ]


with open('README.md') as file:
    long_description = file.read()


setup(
    name='combnet',
    description='Efficient comb filter networks',
    version='0.0.1',
    author='Cameron Churchwell',
    author_email='cameronchurchwell@icloud.com',
    url='https://github.com/CameronChurchwell/combnet',
    install_requires=['accelerate', 'GPUtil', 'torch', 'torchutil', 'yapecs', 'librosa', 'resampy', 'jax', 'pyfluidsynth', 'matplotlib'],
    packages=find_packages(),
    package_data={'combnet': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT',
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
    # ext_modules=modules,
    # cmdclass={'build_ext': BuildExtension}
)
