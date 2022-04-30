from setuptools import setup, find_packages

setup(
    name='damp11113',
    version='2022.4.30.8.12.9',
    license='MIT',
    author='damp11113',
    author_email='damp51252@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/damp11113/damp11113-library',
    description='read on https://github.com/damp11113/damp11113-library',
    install_requires=[
        'Pygments',
        'numba',
        'numpy',
        'pandas',
        'cryptography',
        'pytest',
        'requests',
        'mcstatus',
        'key-generator',
        'opencv-python',
        'pillow',
        'pafy',
        'ffmpeg-python',
        'youtube-dl==2020.12.02',
        'tqdm',
        'pyserial',
        'paho-mqtt',
        'vidstream'
    ]
)