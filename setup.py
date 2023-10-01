from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='damp11113',
    version='2023.10.1.22.0.0',
    license='MIT',
    author='damp11113',
    author_email='damp51252@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/damp11113/damp11113-library',
    description="A Utils library and Easy to using.",
    long_description=long_description,  # Use the content of README.md as the long description
    long_description_content_type='text/markdown',  # Specify that the long description is in Markdown format
    install_requires=[
        "dearpygui",
        "discord.py",
        "iso-639",
        "numpy",
        "scipy",
        "natsort",
        "psutil",
        "Pillow",
        "blend-modes",
        "opencv-python",
        "libscrc",
        "PyAudio",
        "python-vlc",
        "ffmpeg-python",
        "yt-dlp",
        "youtube_dl",
        "pafy",
        "pafy2",
        "tqdm",
        "qrcode",
        "python-barcode",
        "pydub",
        "pyzbar",
        "mcstatus",
        "mcrcon",
        "paho-mqtt",
        "requests",
        "pymata-aio",
        "paramiko",
        "six",
        "key-generator",
        "PyQt5",
        "gTTS"
    ],
    dependency_links=[
        'git+https://github.com/damp11113/pafy2.git#egg=pafy2'
    ]
)
