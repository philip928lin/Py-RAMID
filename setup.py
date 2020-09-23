import setuptools

setuptools.setup(name='py-ramid',
                 version='1.0.1',
                 description='A python package of a Riverware and Agent-based Modeling Interface for Developers.',
                 url='',
                 author='Chung-Yi Lin',
                 author_email='philip928lin@gmail.com',
                 license='NotOpenYet',
                 packages = setuptools.find_packages(),
                 install_requires = ['matplotlib>=3.0.3,<3.1.0', 'numpy>=1.19', 'pandas>=1.1', 'scipy>=1.5', 'tqdm', 'geneticalgorithm','joblib'],
                 zip_safe=False,
                 python_requires='>=3.7')