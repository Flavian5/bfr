from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='bfr',
    version='0.0.3',
    description='Clustering using the BFR algorithm',
    long_description=readme,
    author='Jesper Berglund',
    author_email='jesbergl@kth.se',
    url='https://github.com/jeppeb91/bfr',
    license=license,
    packages=find_packages(exclude='tests'),
    install_requires=requirements
)
