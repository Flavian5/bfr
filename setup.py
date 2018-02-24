from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='bfr',
    version='0.1.0',
    description='Clustering using the BFR algorithm',
    long_description=readme,
    author='Jesper Berglund',
    author_email='jesbergl@kth.se',
    url='TODO',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)