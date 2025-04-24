@echo off

title move pywindows.py to windows exclusive folder
move /src/damp11113/pywindows.py /src/damp11113_windows/

title change urllib3 to 1.26.15
pip install urllib3==1.26.15

title building dist
python setup.py sdist

title uploading to pypi
twine upload -r pypi dist/*

pause