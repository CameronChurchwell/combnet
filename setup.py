from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
    install_requires=['accelerate', 'GPUtil', 'torch', 'torchutil', 'yapecs', 'librosa', 'resampy'],
    packages=find_packages(),
    package_data={'combnet': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT',
    # ext_modules=modules,
    # cmdclass={'build_ext': BuildExtension}
)
