from setuptools import setup, find_packages

setup(
    name='AISample',
    version='1.0.0',
    description='Sample project',
    url='http://github.com/ai-systems/AISample',
    author='AI Systems, University of Manchester',
    author_email='viktor.schlegel@manchester.ac.uk',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    setup_requires=["nose"],
    tests_require=["nose", "coverage"]
)
