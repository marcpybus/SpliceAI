from setuptools import setup
import io


setup(name='spliceai',
      description='SpliceAI: A deep learning-based tool to identify splice variants',
      long_description=io.open('README.md', encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      version='1.3.1',
      author='Kishore Jaganathan',
      author_email='kishorejaganathan@gmail.com',
      license='GPLv3',
      url='https://github.com/illumina/SpliceAI',
      packages=['spliceai'],
      install_requires=['keras==2.0.5',
                        'pyfaidx==0.5.0',
                        'pysam==0.10.0',
                        'numpy==1.14.0',
                        'pandas==0.24.2',
                        'tensorflow==1.15.5'],
      extras_require={'cpu': ['tensorflow==1.15.5'],
                      'gpu': ['tensorflow-gpu==1.15.5']},
      package_data={'spliceai': ['annotations/grch37.txt',
                                 'annotations/grch38.txt',
                                 'models/spliceai1.h5',
                                 'models/spliceai2.h5',
                                 'models/spliceai3.h5',
                                 'models/spliceai4.h5',
                                 'models/spliceai5.h5']},
      entry_points={'console_scripts': ['spliceai=spliceai.__main__:main']},
      test_suite='tests')
