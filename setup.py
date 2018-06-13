from setuptools import setup, find_packages

setup(name='Variable dropout',
      version='0.0.1',
      description='Package provides an implementation of variable dropout method, which can be used to establish the importance of features for any classification or regression model.',
      url='https://github.com/Noctiphobia/variable-dropout/',
      author='Ahmed Abdelkarim, Aleksandra Hernik, Mateusz Mazurkiewicz, Iwona Żochowska',
      author_email='',
      license='MIT',
      packages=find_packages(exclude=['test_variable_dropout']),
      install_requires=[
          'numpy == 1.11.3',
          'scikit-learn == 0.19.1',
          'pandas == 0.22.0',
          'matplotlib == 2.2.0',
          'sphinx_rtd_theme == 0.4.0',
      ])