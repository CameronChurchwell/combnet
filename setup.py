from setuptools import find_packages, setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name='combnet',
    description='Efficient comb filter networks',
    version='0.0.1',
    author='Cameron Churchwell',
    author_email='cameronchurchwell@icloud.com',
    url='https://github.com/CameronChurchwell/combnet',
    install_requires=['accelerate', 'GPUtil', 'torch', 'torchutil', 'yapecs'],
    packages=find_packages(),
    package_data={'combnet': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
