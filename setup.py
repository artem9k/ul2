from setuptools import setup

setup(
      name='ul20',
      version='0.0.1',
      author='Artem Yatsenko',
      packages=['ul2'],
      description='small ul2',
      license='MIT',
      install_requires=[
            'torch',
            'transformers', 
            'accelerate'
      ],
)