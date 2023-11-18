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
import time
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep
from .utils import get_size_unit2

"""
A loader-like context manager

Args:
    desc (str, optional): The loader's description. Defaults to "Loading...".
    end (str, optional): Final print. Defaults to "Done!".
    timeout (float, optional): Sleep time between prints. Defaults to 0.1.
"""
steps1 = ['[   ]', '[-  ]', '[-- ]', '[---]', '[ --]', '[  -]']
steps2 = ['[   ]', '[-  ]', '[ - ]', '[  -]']
steps3 = ['[   ]', '[-  ]', '[-- ]', '[ --]', '[  -]', '[   ]', '[  -]', '[ --]', '[-- ]', '[-  ]']
steps4 = ['[   ]', '[-  ]', '[ - ]', '[  -]', '[   ]', '[  -]', '[ - ]', '[-  ]', '[   ]']
steps5 = ['[   ]', '[  -]', '[ --]', '[---]', '[-- ]', '[-  ]']
steps6 = ['[   ]', '[  -]', '[ - ]', '[-  ]']

class Loading:
    def __init__(self, desc="Loading...", end="[ ✔ ]", timeout=0.1, fail='[ ❌ ]', steps=None):
        self.desc = desc
        self.end = end
        self.timeout = timeout
        self.faill = fail

        self._thread = Thread(target=self._animate, daemon=True)
        if steps is None:
            self.steps = steps1
        else:
            self.steps = steps
        self.done = False
        self.fail = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r{c} {self.desc}", flush=True, end="")
            sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)

    def stopfail(self):
        self.done = True
        self.fail = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.faill}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()

class LoadingProgress:
    def __init__(self, total=100, length=50, fill='█', desc="Loading...", status="", enabuinstatus=True, end="[ ✔ ]", timeout=0.1, fail='[ ❌ ]', steps=None, unit="it", barbackground="-", shortnum=False, shortunitsize=1000, currentshortnum=False, show=True, clearline=True):
        self.desc = desc
        self.end = end
        self.timeout = timeout
        self.faill = fail
        self.total = total
        self.length = length
        self.fill = fill
        self.enbuinstatus = enabuinstatus
        self.status = status
        self.barbackground = barbackground
        self.unit = unit
        self.shortnum = shortnum
        self.shortunitsize = shortunitsize
        self.currentshortnum = currentshortnum
        self.printed = show
        self.clearline = clearline

        self._thread = Thread(target=self._animate, daemon=True)
        if steps is None:
            self.steps = steps1
        else:
            self.steps = steps

        self.currentpercent = 0
        self.current = 0
        self.startime = 0
        self.done = False
        self.fail = False
        self.currentprint = ""

    def start(self):
        self._thread.start()
        self.startime = time.perf_counter()
        return self

    def update(self, i):
        self.current += i

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break

            if self.total != 0 or math.trunc(float(self.currentpercent)) > 100:
                self.currentpercent = ("{0:.1f}").format(100 * (self.current / float(self.total)))
                filled_length = int(self.length * self.current // self.total)
                bar = self.fill * filled_length + self.barbackground * (self.length - filled_length)

                if self.enbuinstatus:
                    elapsed_time = time.perf_counter() - self.startime
                    speed = self.current / elapsed_time if elapsed_time > 0 else 0
                    remaining = self.total - self.current
                    eta_seconds = remaining / speed if speed > 0 else 0
                    elapsed_formatted = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
                    eta_formatted = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
                    if self.shortnum:
                        stotal = get_size_unit2(self.total, '', False, self.shortunitsize, False, '')
                        scurrent = get_size_unit2(self.current, '', False, self.shortunitsize, self.currentshortnum, '')
                    else:
                        stotal = self.total
                        scurrent = self.current

                    if math.trunc(float(self.currentpercent)) > 100:
                        elapsed_time = time.perf_counter() - self.startime
                        elapsed_formatted = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))

                        self.currentprint = f"{c} {self.desc} | {scurrent}/{stotal} | {elapsed_formatted} | {get_size_unit2(speed, self.unit, self.shortunitsize)} | {self.status}"

                    else:
                        self.currentprint = f"{c} {self.desc} | {math.trunc(float(self.currentpercent))}%|{bar}| {scurrent}/{stotal} | {elapsed_formatted}<{eta_formatted} | {get_size_unit2(speed, self.unit, self.shortunitsize)} | {self.status}"
                else:
                    if self.shortnum:
                        stotal = get_size_unit2(self.total, '', False, self.shortunitsize, False, '')
                        scurrent = get_size_unit2(self.current, '', False, self.shortunitsize, self.currentshortnum, '')
                    else:
                        stotal = self.total
                        scurrent = self.current
                    self.currentprint = f"{c} {self.desc} | {math.trunc(float(self.currentpercent))}%|{bar}| {scurrent}/{stotal} | {self.status}"
            else:
                elapsed_time = time.perf_counter() - self.startime
                elapsed_formatted = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))

                self.currentprint = f"{c} {self.desc} | {elapsed_formatted} | {self.status}"

            if self.printed:
                print(f"\r{self.currentprint}", flush=True, end="")

            sleep(self.timeout)

            if self.printed and self.clearline:
                cols = get_terminal_size((80, 20)).columns
                print("\r" + " " * cols, end="", flush=True)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)

    def stopfail(self):
        self.done = True
        self.fail = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.faill}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()