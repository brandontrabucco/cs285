"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'tensorflow-gpu==2.0.0-rc0',
    'numpy',
    'mujoco-py',
    'gym[all]']


setup(
    name='cs285', version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('cs285')],
    description='Official Deep RL Framework by Brandon Trabucco.')
