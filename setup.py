from setuptools import setup, find_packages
import launcher

setup(
    name='launcher',
    version=str(launcher.__VERSION__),
    py_modules=['launcher'],
    license="MIT",
)
