from setuptools import setup, find_packages

setup(
    name='damp11113',
    version='2022.11.14.8.14.3',
    license='MIT',
    author='damp11113',
    author_email='damp51252@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/damp11113/damp11113-library',
    description='A Utils library and Easy to using.',
    install_requires=[
        "Pygments",
        "gTTS",
        "playsound",
        "PyQt5",
        "natsort",
        "requests",
        "psutil",
        "python-vlc",
        "pafy",
        "pycocotools @ git+https://github.com/damp11113/pafy2.git",
        "ffmpeg-python",
        "opencv-python",
        "tqdm",
        "numpy",
        "qrcode",
        "barcode",
        "Pillow",
        "yt-dlp",
        "pyzbar",
        "mcstatus",
        "mcrcon",
        "paho-mqtt",
        "youtube_dl",
        "key-generator"
    ]
)
