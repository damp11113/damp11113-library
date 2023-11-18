"""
damp11113-library - A Utils library and Easy to use. For more info visit https://github.com/damp11113/damp11113-library/wiki
Copyright (C) 2021-2023 damp11113 (MIT)

Visit https://github.com/damp11113/damp11113-library

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import platform

def pyversion(fullpython=False, fullversion=False, tags=False, date=False, compiler=False, implementation=False, revision=False):
    if fullpython:
        return f'python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} {sys.version_info.releaselevel} {platform.python_build()[0]} {platform.python_build()[1]} {platform.python_compiler()} {platform.python_implementation()} {platform.python_revision()}'
    if fullversion:
        return f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
    if tags:
        return platform.python_build()[0]
    if date:
        return platform.python_build()[1]
    if compiler:
        return platform.python_compiler()
    if implementation:
        return platform.python_implementation()
    if revision:
        return platform.python_revision()
    return f'{sys.version_info.major}.{sys.version_info.minor}'

def osversion(fullos=False, fullversion=False, type=False, cuser=False, processor=False):
    if fullos:
        return f'{platform.node()} {platform.platform()} {platform.machine()} {platform.architecture()[0]} {platform.processor()}'
    if fullversion:
        return f'{platform.system()} {platform.version()}'
    if type:
        return platform.architecture()[0]
    if cuser:
        return platform.node()
    if processor:
        return platform.processor()

    return platform.release()