from setuptools import setup, find_packages

setup(
    name='strypy',
    version='0.1',
    packages=find_packages(exclude=["examples"]),
    install_requires=["obspy"],
    python_requires='>=3.11',
)
