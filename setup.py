from setuptools import setup, find_packages

setup(
    name='motpy',
    version='0.1.0',    
    description=' Python implementations of multi-object tracking algorithms',
    url='https://github.com/ShaneFlandermeyer/MOTpy',
    author='Shane Flandermeyer',
    author_email='shaneflandermeyer@gmail.com',
    license='MIT',
    packages=find_packages(exclude=("test",)),
    install_requires=[
      'numpy',
      'scipy',
      'matplotlib',
      'pytest',                     
    ],

)
