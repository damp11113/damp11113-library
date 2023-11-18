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
import math
import operator
import time, sys
from functools import reduce
from time import sleep
from datetime import datetime
from threading import Thread
from .randoms import rannum
from .file import removefile, readfile, writefile2, createfile
from inspect import getmembers, isfunction
import ctypes
from gtts import gTTS
from playsound import playsound
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

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

def typing(text, speed=0.3):
    for character in text:
        sys.stdout.write(character)
        sys.stdout.flush()
        time.sleep(speed)

def timestamp():
    now = datetime.now()
    return datetime.timestamp(now)

def full_cpu(min=100, max=10000, speed=0.000000000000000001):
    _range = rannum(min, max)
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
    ra = rannum(min, max)
    for i in range(ra):
        createfile('test.txt')
        writefile2('test.txt', 'test')
        readfile('test.txt')
        removefile('test.txt')
        sleep(speed)
        print(f'{i}/{ra}')


def mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

def tts(text, lang, play=True, name='tts.mp3', slow=False):
    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save(name)
    if play:
        playsound(name)
        removefile(name)

textt = """Unhandled exception has occurred in your application.If you click\nContinue,the application will ignore this error and attempt to continue.\nIf you click Quit,the application will close immediately.\n"""

def emb(info, details=None, text=textt, title='python'):
    app = QApplication(sys.argv)
    msg = QMessageBox()

    msg.setEscapeButton(QMessageBox.Close)
    msg.setIcon(QMessageBox.Critical)

    msg.setText(text)
    msg.setInformativeText(info)
    msg.setWindowTitle(title)

    if details is not None:
        msg.setDetailedText(details)

    msg.addButton(QPushButton('Continue'), QMessageBox.YesRole)
    msg.addButton(QPushButton('Quit'), QMessageBox.NoRole)

    retval = msg.exec_()

    if retval == 0:
        return True
    else:
        exit()

def emb2(info, details=None, text=textt, title='python'):
    msg = QMessageBox()

    msg.setEscapeButton(QMessageBox.Close)
    msg.setIcon(QMessageBox.Critical)

    msg.setText(text)
    msg.setInformativeText(info)
    msg.setWindowTitle(title)

    if details is not None:
        msg.setDetailedText(details)

    msg.addButton(QPushButton('Continue'), QMessageBox.YesRole)
    msg.addButton(QPushButton('Quit'), QMessageBox.NoRole)

    retval = msg.exec_()

    if retval == 0:
        return True
    else:
        sys.exit()

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
            return f"{bytes:.2f} {unit}B/s"
        bytes /= 1024

def get_size_unit2(number, unitp, persec=True, unitsize=1024, decimal=True, space=" "):
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if number < unitsize:
            if decimal:
                num = f"{number:.2f}"
            else:
                num = math.trunc(number)

            if persec:
                return f"{num}{space}{unit}{unitp}/s"
            else:
                return f"{num}{space}{unit}{unitp}"
        number /= unitsize

def get_percent_completed(current, total):
    return round((current * 100) / total)

def textonumber(text):
    l = []
    tl = list(text)
    for i in tl:
        l.append(ord(i))
    return ''.join(str(v) for v in l)


def Amap(x, in_min, in_max, out_min, out_max):
    try:
        return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
    except:
        return 0

def get_remaining_time(current, total):
    if total == 0:
        return None
    else:
        totalMin = 1440 - 60 * current - total
        hoursRemaining = totalMin // 60
        minRemaining = totalMin % 60
        return (hoursRemaining, minRemaining)

def isAscii(string):
    return reduce(operator.and_, [ord(x) < 256 for x in string], True)

def get_all_func_in_module(module):
    func = []
    for i in getmembers(module, isfunction):
        func.append(i[0])
    return func
    
def get_format_time(sec):
    if sec >= 31557600: # years
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        y, d = divmod(d, 365)
        return f"{y}y {d}d {h}h {m}m {s}s"
    elif sec >= 2628002: # months
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        mo, d = divmod(d, 30)
        return f"{mo}mo {d}d {h}h {m}m {s}s"
    elif sec >= 604800:# weeks
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        w, d = divmod(d, 7)
        return f"{w}w {d}d {h}h {m}m {s}s"
    elif sec >= 86400: # days
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        return f"{d}d {h}h {m}m {s}s"
    elif sec >= 3600:# hours
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        return f"{h}h {m}m {s}s"
    elif sec >= 60: # minutes
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return f"{m}m {s}s"
    else:
        return f"{sec}s"

def get_format_time2(sec):
    if sec >= 31557600: # years
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        y, d = divmod(d, 365)
        return f"{y}y {d}d {str(h).zfill(2)}:{str(m).zfill(2)}:{str(sec).zfill(2)}"
    elif sec >= 2628002: # months
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        mo, d = divmod(d, 30)
        return f"{mo}mo {d}d {str(h).zfill(2)}:{str(m).zfill(2)}:{str(sec).zfill(2)}"
    elif sec >= 604800:# weeks
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        w, d = divmod(d, 7)
        return f"{w}w {d}d {str(h).zfill(2)}:{str(m).zfill(2)}:{str(sec).zfill(2)}"
    elif sec >= 86400: # days
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        return f"{d}d {str(h).zfill(2)}:{str(m).zfill(2)}:{str(sec).zfill(2)}"
    elif sec >= 3600:# hours
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        return f"{str(h).zfill(2)}:{str(m).zfill(2)}:{str(sec).zfill(2)}"
    elif sec >= 60: # minutes
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return f"{str(m).zfill(2)}:{str(sec).zfill(2)}"
    else:
        return f"00:{str(sec).zfill(2)}"

def addStringEveryN(original_string, add_string, n):
    # Input validation
    if not isinstance(original_string, str) or not isinstance(add_string, str):
        raise TypeError("Both original_string and add_string must be strings.")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")

    chunks = [original_string[i:i+n] for i in range(0, len(original_string), n)]
    result = add_string.join(chunks)
    return result

def findStringDifferencesInList(list1, list2):
    # Convert the input lists to sets
    set1 = set(list1)
    set2 = set(list2)

    # Find the differences between the sets
    difference = list(set1.symmetric_difference(set2))

    return difference

def replaceEnterWithCrlf(input_string):
    if '\n' in input_string:
        input_string = input_string.replace('\n', '\r\n')
    return input_string

def scrollTextBySteps(text, scrollstep, scrollspace=10):
    if len(text) < scrollspace:
        raise ValueError("text is shorter than scroll space")
    if len(text) < scrollstep:
        raise ValueError("text is shorter than scroll step")

    scrolled_text = text
    scrolled = ""

    for _ in range(0, scrollstep+1):
        scrolled = f"{scrolled_text[:scrollspace]:<{scrollspace}}"

        # Shift the text by one character to the right
        scrolled_text = scrolled_text[1:] + scrolled_text[0]

        # Add a space at the end if the text length is less than 8 characters
        if len(scrolled_text) < scrollspace:
            scrolled_text += " "

    return scrolled

def calculate_pi(n):
    pi = 0.0
    for k in range(n):
        pi += 1.0 / (16 ** k) * (
            4.0 / (8 * k + 1) - 2.0 / (8 * k + 4) - 1.0 / (8 * k + 5) - 1.0 / (8 * k + 6)
        )
    return pi

def findMedian(list_numbers: list):
    list_numbers.sort()
    length = len(list_numbers)
    if length % 2 == 0:
        return (list_numbers[length // 2 - 1] + list_numbers[length // 2]) / 2
    else:
        return list_numbers[length // 2]

def stemLeafPlot(data):
    stems = {}
    result = ""

    for num in data:
        stem = num // 10
        leaf = num % 10
        if stem not in stems:
            stems[stem] = []
        stems[stem].append(leaf)

    for stem, leaves in sorted(stems.items()):
        result += f"{stem} | {' '.join(map(str, sorted(leaves)))}\n"

    return result

def dotPlot(data, dot=". ", showlable=True):
    max_value = max(data)
    dot_plot = ''

    for i in range(max_value, 0, -1):
        row = ''
        for value in data:
            if value >= i:
                row += dot  # Use a dot to represent the value
            else:
                row += ' ' * len(dot)  # Use empty space if the value is lower
        dot_plot += row + '\n'

    if showlable:
        x_axis_labels = ' '.join(str(i) for i in range(1, len(data) + 1))
        dot_plot += x_axis_labels + '\n'

    return dot_plot

def texttable(data):
    table_str = ""

    # Calculate the width of each column based on the maximum length of data in each column
    col_width = [max(len(str(row[i])) for row in data) for i in range(len(data[0]))]

    # Create table header
    for i in range(len(data[0])):
        table_str += str(data[0][i]).ljust(col_width[i]) + ' '
    table_str += '\n'

    # Create separator
    for width in col_width:
        table_str += '-' * width + ' '
    table_str += '\n'

    # Create table content
    for row in data[1:]:
        for i in range(len(row)):
            table_str += str(row[i]).ljust(col_width[i]) + ' '
        table_str += '\n'

    return table_str