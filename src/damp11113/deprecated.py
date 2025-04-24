"""
This file is for deprecated functions and classes.

damp11113-library - A Utils library and Easy to use. For more info visit https://github.com/damp11113/damp11113-library/wiki
Copyright (C) 2021-present damp11113 (MIT)

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

import subprocess
import sys
import time
from datetime import datetime
from threading import Thread
import requests
from gtts import gTTS
from mcrcon import MCRcon
from mcstatus import JavaServer
import vlc
import pafy
import warnings
from playsound import playsound

from .randoms import rannum
from .minecraft import mcstatus_exception

warnings.warn("Deprecated functions and classes are delete in future.", DeprecationWarning)

class rcon_exception(Exception):
    pass


class vlc_player:
    def __init__(self):
        warnings.warn("vlc_player is deprecated, use python-vlc instead", DeprecationWarning)

        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()

    def load(self, file_path):
        self.media = self.instance.media_new(file_path)
        self.player.set_media(self.media)
        return f"Loading {file_path}"

    def play(self):
        self.player.play()
        return f"Playing {self.media.get_mrl()}"

    def pause(self):
        self.player.pause()
        return f"Pausing {self.media.get_mrl()}"

    def stop(self):
        self.player.stop()
        return f"Stopping {self.media.get_mrl()}"

    def get_position(self):
        return self.player.get_position()

    def set_position(self, position):
        self.player.set_position(position)
        return f"Setting position to {position}"

    def get_state(self):
        return self.player.get_state()

    def get_length(self):
        return self.media.get_duration()

    def get_time(self):
        return self.player.get_time()

    def set_time(self, time):
        self.player.set_time(time)
        return f"Setting time to {time}"

    def get_rate(self):
        return self.player.get_rate()

    def set_rate(self, rate):
        self.player.set_rate(rate)
        return f"Setting rate to {rate}"

    def get_volume(self):
        return self.player.audio_get_volume()

    def set_volume(self, volume):
        self.player.audio_set_volume(volume)
        return f"Setting volume to {volume}"

    def get_mute(self):
        return self.player.audio_get_mute()

    def set_mute(self, mute):
        self.player.audio_set_mute(mute)
        return f"Setting mute to {mute}"

    def get_chapter(self):
        return self.player.get_chapter()

    def set_chapter(self, chapter):
        self.player.set_chapter(chapter)
        return f"Setting chapter to {chapter}"

    def get_chapter_count(self):
        return self.media.get_chapter_count()

    def get_title(self):
        return self.player.get_title()

    def set_title(self, title):
        self.player.set_title(title)
        return f"Setting title to {title}"

    def get_title_count(self):
        return self.media.get_title_count()

    def get_video_track(self):
        return self.player.video_get_track()

    def set_video_track(self, track):
        self.player.video_set_track(track)
        return f"Setting video track to {track}"

    def get_video_track_count(self):
        return self.media.get_video_track_count()

    def get_audio_track(self):
        return self.player.audio_get_track()

    def set_audio_track(self, track):
        self.player.audio_set_track(track)
        return f"Setting audio track to {track}"

    def get_audio_track_count(self):
        return self.media.get_audio_track_count()

    def get_spu_track(self):
        return self.player.video_get_spu()

    def set_spu_track(self, track):
        self.player.video_set_spu(track)
        return f"Setting subtitle track to {track}"

    def get_spu_track_count(self):
        return self.media.get_spu_track_count()

    def get_chapter_description(self, chapter):
        return self.media.get_chapter_description(chapter)

    def toggle_fullscreen(self):
        self.player.toggle_fullscreen()
        return f"Toggling fullscreen"

    def get_fullscreen(self):
        return self.player.get_fullscreen()

    def get_video_resolution(self):
        return (self.player.video_get_width(), self.player.video_get_height())

    def get_fps(self):
        return self.player.get_fps()

class youtube_stream:
    def __init__(self, url):
        warnings.warn("youtube_stream is deprecated, use pafy instead", DeprecationWarning)
        self.stream = pafy.new(url)

    def video_stream(self, resolution=None):
        if resolution is None:
            best = self.stream.getbestvideo()
        else:
            best = self.stream.getbestvideo().resolution(resolution)
        return best.url

    def audio_stream(self):
        best = self.stream.getbestaudio()
        return best.url

    def best_stream(self, resolution=None):
        if resolution is None:
            best = self.stream.getbest()
        else:
            best = self.stream.getbest().resolution(resolution)
        return best.url

    def get_title(self):
        return self.stream.title

    def get_dec(self):
        return self.stream.description

    def get_length(self):
        return self.stream.length

    def get_thumbnail(self):
        return self.stream.thumb

    def get_author(self):
        return self.stream.author

    def get_likes(self):
        return self.stream.likes

    def get_dislikes(self):
        return self.stream.dislikes

    def get_views(self):
        return self.stream.viewcount

class mcserver:
    def __init__(self, server='server.jar', java='java', ramuse='1024', nogui=True, noguipp=False):
        warnings.warn("mcserver is deprecated", DeprecationWarning)
        self.serverf = server
        self.java = java
        self.ramuse = ramuse
        self.nogui = nogui
        self.noguipp = noguipp

    def start(self):
        if self.nogui:
            if self.noguipp:
                self.s = subprocess.Popen(
                    f'{self.java} -Xms{self.ramuse}M -Xmx{self.ramuse}M -jar {self.serverf} --nogui', shell=True,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                self.s = subprocess.Popen(
                    f'{self.java} -Xms{self.ramuse}M -Xmx{self.ramuse}M -jar {self.serverf} nogui', shell=True,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            self.s = subprocess.Popen(f'{self.java} -Xms{self.ramuse}M -Xmx{self.ramuse}M -jar {self.serverf}',
                                      shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def stop(self):
        self.command('stop')
        self.s.communicate()

    def command(self, command):
        self.s.stdin.write(command + '\n')

    def log(self):
        return self.s.stdout.read()

def mcstatusplayerparse(sample):
    listplayer = []
    for p in sample:
        listplayer.append(p.name)
    return listplayer

class mcstatus:
    def __init__(self, ip):
        warnings.warn("mcstatus is deprecated, use mcstatus instead.", DeprecationWarning)
        self.server = JavaServer.lookup(ip)

    def raw(self):
        try:
            return self.server.status().raw
        except Exception as e:
            raise mcstatus_exception(f"raw mc status error: {e}")

    def players(self):
        try:
            return self.server.status().players
        except Exception as e:
            raise mcstatus_exception(f"players mc status error: {e}")

    def favicon(self):
        try:
            return self.server.status().favicon
        except Exception as e:
            raise mcstatus_exception(f"favicon mc status error: {e}")

    def description(self):
        try:
            return self.server.status().description
        except Exception as e:
            raise mcstatus_exception(f"description mc status error: {e}")

    def version(self):
        try:
            return self.server.status().version
        except Exception as e:
            raise mcstatus_exception(f"version mc status error: {e}")

    def ping(self):
        try:
            return self.server.ping()
        except Exception as e:
            raise mcstatus_exception(f"ping mc status error: {e}")

    def query_raw(self):
        try:
            return self.server.query().raw
        except Exception as e:
            raise mcstatus_exception(f"query raw mc status error: {e}")

    def query_players(self):
        try:
            return self.server.query().players
        except Exception as e:
            raise mcstatus_exception(f"query players mc status error: {e}")

    def query_map(self):
        try:
            return self.server.query().map
        except Exception as e:
            raise mcstatus_exception(f"query map mc status error: {e}")

    def query_motd(self):
        try:
            return self.server.query().motd
        except Exception as e:
            raise mcstatus_exception(f"query motd mc status error: {e}")

    def query_software(self):
        try:
            return self.server.query().software
        except Exception as e:
            raise mcstatus_exception(f"query software mc status error: {e}")

class Rcon:
    def __init__(self, ip, password, port=25575, tls=0, timeout=5):
        warnings.warn("Rcon is deprecated, use MCRcon instead.", DeprecationWarning)

        self.ip = ip
        self.port = port
        self.password = password
        self.rcon = MCRcon(host=ip, port=port, password=password, tlsmode=tls, timeout=timeout)

    def connect(self):
        try:
            self.rcon.connect()
        except Exception as e:
            raise rcon_exception(f"rcon connect error: {e}")

    def send(self, command):
        try:
            return self.rcon.command(command)
        except Exception as e:
            raise rcon_exception(f"rcon send error: {e}")

    def disconnect(self):
        try:
            self.rcon.disconnect()
        except Exception as e:
            raise rcon_exception(f"rcon disconnect error: {e}")


def distrochecker(giftcode):
    warnings.warn("DistroChecker is deprecated due to violating Discord ToS", DeprecationWarning)

    r = requests.get(f'https://discordapp.com/api/v9/entitlements/gift-codes/{giftcode}?with_application=false&with_subscription_plan=true')
    if r == 200:
        return ('ok', giftcode)
    else:
        return ('error', giftcode)

def grade(score: int):
    warnings.warn("grade is deprecated. Just coding with yourself!", DeprecationWarning)
    if 0 >= score >= 100:
        return '0 - 100 only'

    if score == 100:
        return "Perfect"
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
    warnings.warn("clock is deprecated. Use datetime module instead.", DeprecationWarning)

    x = datetime.now()
    clock = x.strftime(display) #"%z %A %d %B %Y  %p %H:%M:%S"
    return clock

class BooleanArgs:
    def __init__(self, args):
        warnings.warn("BooleanArgs is deprecated.", DeprecationWarning)
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
    warnings.warn("typing is deprecated. It useless.", DeprecationWarning)

    for character in text:
        sys.stdout.write(character)
        sys.stdout.flush()
        time.sleep(speed)

def timestamp(timezone=None):
    warnings.warn("timestamp is deprecated. Use datetime module instead.", DeprecationWarning)

    return datetime.timestamp(datetime.now(timezone))

def full_cpu(min=100, max=10000, speed=0.000000000000000001):
    warnings.warn("full_cpu is deprecated. It useless.", DeprecationWarning)

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
        time.sleep(speed)

def tts(text, lang, play=True, name='tts.mp3', slow=False):
    warnings.warn("tts is deprecated. Use gTTS directly instead.", DeprecationWarning)

    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save(name)
    if play:
        playsound(name)

