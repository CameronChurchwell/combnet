from setuptools import find_packages, setup


with open('README.md') as file:
    long_description = file.read()


# TODO - replace with details of your project
setup(
    name='NAME',
    description='DESCRIPTION',
    version='0.0.1',
    author='AUTHOR',
    author_email='EMAIL',
    url='https://github.com/USERNAME/NAME',
    install_requires=['accelerate', 'GPUtil', 'torch', 'torchutil', 'yapecs'],
    packages=find_packages(),
    package_data={'NAME': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
