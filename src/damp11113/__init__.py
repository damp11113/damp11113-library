import time, sys
from time import sleep
from pygments import console
from datetime import datetime
import math
from threading import Thread
import platform
import damp11113.randoms as rd
import damp11113.file as file
import ctypes
import os
from gtts import gTTS
from playsound import playsound
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import numpy as np
from base64 import b64decode
import natsort

class time_exception(Exception):
    pass

def grade(number):
    score = int(number)
    if score == 100:
        return "Perfect"
    elif score >= 100:
        return '0 - 100 only'
    elif score >= 95:
        return "A+"
    elif score >= 90:
        return "A"
    elif score >= 85:
        return "B+"
    elif score >= 80:
        return "B"
    elif score >= 75:
        return "C+"
    elif score >= 70:
        return "C"
    elif score >= 65:
        return "D+"
    elif score >= 60:
        return "D"
    elif score >= 55:
        return "E+"
    elif score >= 50:
        return "E"

    else:
        return "F"

def clock(display="%z %A %d %B %Y - %H:%M:%S"):
    x = datetime.now()
    clock = x.strftime(display) #"%z %A %d %B %Y  %p %H:%M:%S"
    return clock

def check(list, use):
    if use in list:
        print(f'[{console.colorize("green", "✔")}] {use}')
        rech = True

    else:
        print(f'[{console.colorize("red", "❌")}] {use}')
        rech = False
    return rech

def timestamp2date(timestamp, display='%Y-%m-%d %H:%M:%S'):
    return datetime.fromtimestamp(timestamp).strftime(display)

class BooleanArgs:
    def __init__(self, args):
        self._args = {}
        self.all = False

        for arg in args:
            arg = arg.lower()

            if arg == "-" or arg == "!*":
                self.all = False
                self._args = {}

            if arg == "+" or arg == "*":
                self.all = True

            if arg.startswith("!"):
                self._args[arg.strip("!")] = False

            else:
                self._args[arg] = True

    def get(self, item):
        return self.all or self._args.get(item, False)

    def __getattr__(self, item):
        return self.get(item)

def sec2mph(sec):
    return (sec * 2.2369)

def str2bin(s):
    return ''.join(format(ord(x), '08b') for x in s)

def bin2str(b):
    return ''.join(chr(int(b[i:i+8], 2)) for i in range(0, len(b), 8))

def typing(text, speed=0.3):
    for character in text:
        sys.stdout.write(character)
        sys.stdout.flush()
        time.sleep(speed)

def timestamp():
    now = datetime.now()
    return datetime.timestamp(now)

def list2str(list_):
    return '\n'.join(list_)

def str2list(string):
    return string.split('\n')

def str2int(string):
    return int(string)

def byte2str(b, decode='utf-8'):
    return b.decode(decode)

def sort_files(file_list ,reverse=False):
    flist = []
    for file in file_list:
        flist.append(file)
    return natsort.natsorted(flist, reverse=reverse)


def base64decode(base):
    return b64decode(base)

def full_cpu(min=100, max=10000, speed=0.000000000000000001):
    _range = rd.rannum(min, max)
    class thread_class(Thread):
        def __init__(self, name, _range):
            Thread.__init__(self)
            self.name = name
            self.range = _range
        def run(self):
            for i in range(self.range):
                print(f'{self.name} is running')
    for i in range(_range):
        name = f'Thread {i}/{_range}'
        thread = thread_class(name, _range)
        thread.start()
        sleep(speed)

def full_disk(min=100, max=10000,speed=0.000000000000000001):
    ra = rd.rannum(min, max)
    for i in range(ra):
        file.createfile('test.txt')
        file.writefile2('test.txt', 'test')
        file.readfile('test.txt')
        file.removefile('test.txt')
        sleep(speed)
        print(f'{i}/{ra}')

def copyright():
    print('Copyright (c) 2021-2022 damp11113'
          '\nAll rights reserved.'
          '\n '
          '\nMIT License'
          '\n '
          '\nPermission is hereby granted, free of charge, to any person obtaining a copy'
          '\nof this software and associated documentation files (the "Software"), to deal'
          '\nin the Software without restriction, including without limitation the rights'
          '\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell'
          '\ncopies of the Software, and to permit persons to whom the Software is'
          '\nfurnished to do so, subject to the following conditions:'
          '\n '
          '\nThe above copyright notice and this permission notice shall be included in all'
          '\ncopies or substantial portions of the Software.'
          '\n '
          '\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR'
          '\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,'
          '\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE'
          '\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER'
          '\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,'
          '\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE'
          '\nSOFTWARE.')
    return 'Copyright (c) 2021-2022 damp11113 All rights reserved. (MIT License)'

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

def mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

def tts(text, lang, play=True, name='tts.mp3', slow=False):
    tts = gTTS(text=text, lang=lang, slow=slow, )
    tts.save(name)
    if play:
        playsound(name)
        file.removefile(name)

textt = """Unhandled exception has occurred in your application.If you click\nContinue,the application will ignore this error and attempt to continue.\nIf you click Quit,the application will close immediately.\n"""

def emb(info, details=None, text=textt, title='python'):
    app = QApplication(sys.argv)
    msg = QMessageBox()
    # remove title icon
    msg.setEscapeButton(QMessageBox.Close)
    msg.setWindowIcon(QIcon())
    msg.setIcon(QMessageBox.Critical)

    msg.setText(text)
    msg.setInformativeText(info)
    msg.setWindowTitle(title)
    if not details is None:
        msg.setDetailedText(details)
    msg.addButton(QPushButton('Continue'), QMessageBox.YesRole)
    msg.addButton(QPushButton('Quit'), QMessageBox.NoRole)
    retval = msg.exec_()

    if retval == 0:
        return True
    else:
        return False

def emb2(info, details=None, text=textt, title='python'):
    msg = QMessageBox()
    # remove title icon
    msg.setEscapeButton(QMessageBox.Close)
    msg.setWindowIcon(QIcon())
    msg.setIcon(QMessageBox.Critical)

    msg.setText(text)
    msg.setInformativeText(info)
    msg.setWindowTitle(title)
    if not details is None:
        msg.setDetailedText(details)
    msg.addButton(QPushButton('Continue'), QMessageBox.YesRole)
    msg.addButton(QPushButton('Quit'), QMessageBox.NoRole)
    retval = msg.exec_()

    if retval == 0:
        return True
    else:
        return False

class Queue:
    def __init__(self, queue):
        self.queue = queue
    def put(self, item):
        self.queue.append(item)
    def get(self):
        r = self.queue[0]
        self.queue.pop(0)
        return r

def get_size_unit(bytes):
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}B"
        bytes /= 1024

def get_percent_completed(current, total):
    return round((current * 100) / total)