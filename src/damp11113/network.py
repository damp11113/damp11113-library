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

import socket
import traceback
from tqdm import tqdm
import webbrowser
import paho.mqtt.client as mqtt
import time
from .file import *
import youtube_dl
from .convert import bin2str, byte2str, str2bin
import yt_dlp as youtube_dl2
import re
import requests
from .utils import get_size_unit, emb
from .processbar import LoadingProgress, steps5

class vc_exception(Exception):
    pass

class line_api_exception(Exception):
    pass

class ip_exeption(Exception):
    pass

class receive_exception(Exception):
    pass

class send_exception(Exception):
    pass


def youtube_search(search, randomsearch=False):
    formatUrl = requests.get(f'https://www.youtube.com/results?search_query={search}')
    search_result = re.findall(r'watch\?v=(\S{11})', formatUrl.text)

    if randomsearch:
        return f"https://www.youtube.com/watch?v={search_result[0]}"
    else:
        return search_result

#---------------------------ip--------------------------------

def ip_port_check(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, port))
        rech = f'{ip}:{port} is connected'
    except socket.error as e:
        raise ip_exeption(f'{ip}:{port} is disconnected')
    return rech

attack_num = 0

def http_ddos_attack(target): #beta
    while True:
        try:
            d = requests.get(target)
            global attack_num
            attack_num += 1
            print(f'[{attack_num}] {target} is connected')
            d.close()
        except:
            print(f'[-] ddos attack is stop because {target} is disconnected')
            pass

#-------------------------download---------------------------

def loadfile(url, filename):
    progress = LoadingProgress(desc=f'loading file from {url}', steps=steps5, unit="B", shortunitsize=1024, shortnum=True)
    progress.start()
    try:
        progress.desc = f'Downloading {filename} from {url}'
        progress.status = "Connecting..."
        r = requests.get(url, stream=True)
        progress.desc = f'Downloading {filename} from {url}'
        progress.status = "Starting..."
        tsib = int(r.headers.get('content-length', 0))
        bs = 1024
        progress.total = tsib
        progress.desc = f'Downloading {filename} from {url}'
        progress.status = "Downloading..."
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=bs):
                progress.update(len(chunk))
                f.write(chunk)
        progress.status = "Downloaded"
        progress.end = f"[ ✔ ] Downloaded {filename} from {url}"
        progress.stop()
    except Exception as e:
        progress.status = "Error"
        progress.faill = f"[ ❌ ] Failed to download {filename} from {url} | " + str(e)
        progress.stopfail()
        emb(str(e), traceback.print_exc())

def installpackage(package):
    os.system(f'title install {package}')
    os.system(f'pip install {package}')

ydl_optss = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    'outtmpl': '%(title)s.%(ext)s'
}

def ytload(url, ydl_opts=ydl_optss):
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except:
        print("ytload error")
        pass

def ytload2(url, ydl_opts=ydl_optss):
    try:
        with youtube_dl2.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except:
        print("ytload error")
        pass
#----------------------------read------------------------------

def readtextweb(url):
    try:
        r = requests.get(url)
        return r.text
    except Exception as e:
        print("read error", e)

#-----------------------------open-----------------------------

def openurl(url):
    webbrowser.open(url)

#-----------------------------send-----------------------------

def sendtext(url, text):
    try:
        return requests.post(url, data=text).status_code
    except Exception as e:
        raise send_exception(f'send error: {e}')

def sendfile(url, file):
    try:
        requests.post(url, files=file)
    except Exception as e:
        raise send_exception(f'send error: {e}')

def mqtt_publish(topic, message, port=1883, host="localhost"):
    try:
        client = mqtt.Client()
        client.connect(host, port, 60)
        client.publish(topic, message)
        client.disconnect()
    except Exception as e:
        raise send_exception(f'send error: {e}')

def tcp_send(host, port, message):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.sendall(bytes(message, 'utf-8'))
        s.close()
        print(f"tcp send to {host}:{port}")
    except Exception as e:
        raise send_exception(f'send error: {e}')

def udp_send(host, port, message):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((host, port))
        s.sendall(bytes(message, "utf-8"))
        s.close()
        print(f"udp send to {host}:{port}")
    except Exception as e:
        raise send_exception(f'send error: {e}')

def binary_send(host, port, message):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.sendall(bytes(str2bin(message), 'utf-8'))
        s.close()
        print(f"binary send to {host}:{port}")
    except Exception as e:
        raise send_exception(f'send error: {e}')

def file_send(host, port, file, buffsize=4096, speed=0.0000001):
    try:
        filesize = sizefile(file)
        s = socket.socket()
        s.connect((host, port))
        s.send(f"{file}{filesize}".encode())
        progress_bar = tqdm(total=filesize, unit='B', unit_scale=True, desc=f'Sending {file}')
        with open(file, 'rb') as f:
            while True:
                data = f.read(buffsize)
                if not data:
                    break
                s.sendall(data)
                progress_bar.update(len(data))
                time.sleep(speed)
        s.close()
        progress_bar.close()
    except Exception as e:
        raise send_exception(f'send error: {e}')

#-----------------------------receive--------------------------

def mqtt_subscribe(topic, port=1883, host="localhost"):
    try:
        client = mqtt.Client()
        client.connect(host, port)
        mes = client.subscribe(topic)
        client.disconnect()
        return mes
    except Exception as e:
        raise receive_exception(f'receive error: {e}')

def tcp_receive(host, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(1)
        conn, addr = s.accept()
        data = conn.recv(1024)
        conn.close()
        print(f"tcp receive from {host}:{port}")
        return byte2str(data)
    except Exception as e:
        raise receive_exception(f'receive error: {e}')

def udp_receive(host, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((host, port))
        data, addr = s.recvfrom(1024)
        s.close()
        print(f"udp receive from {host}:{port}")
        return byte2str(data)
    except Exception as e:
        raise receive_exception(f'receive error: {e}')

def binary_receive(host, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(1)
        conn, addr = s.accept()
        data = bin2str(conn.recv(1024))
        conn.close()
        print(f"binary receive from {host}:{port}")
        return byte2str(data)
    except Exception as e:
        raise receive_exception(f'receive error: {e}')

def file_receive(host, port, buffsize=4096, speed=0.0000001):
    try:
        s = socket.socket()
        s.bind((host, port))
        s.listen(5)
        conn, addr = s.accept()
        received = conn.recv(buffsize)
        filename = received.decode()
        filesize = int(byte2str(conn.recv(1024), decode='utf-16'))
        progress_bar = tqdm(total=filesize, unit='B', unit_scale=True, desc=f'Receiving {filename}')
        with open(filename, 'wb') as f:
            while True:
                data = conn.recv(buffsize)
                if not data:
                    break
                f.write(data)
                progress_bar.update(len(data))
                time.sleep(speed)
        progress_bar.close()
        conn.close()
    except Exception as e:
        raise receive_exception(f'receive error: {e}')

#-----------------------------run-----------------------------

def runngrok(type, ip, port):
    os.system(f"start ngrok {type} {ip}:{port}")

#---------flask------------------

def flask_run(url, port):
    os.system(f"flask run --host {url} --port {port}")

def flask_run_debug(url, port):
    os.system(f"flask run --host {url} --port {port} --debug")

def flask_run_ssl(url, port, cert, key):
    os.system(f"flask run --host {url} --port {port} --ssl-cert {cert} --ssl-key {key}")

def flask_run_debug_ssl(url, port, cert, key):
    os.system(f"flask run --host {url} --port {port} --debug --ssl-cert {cert} --ssl-key {key}")

#----webshell------------------

def runwebshell(url):
    os.system(f"webshell {url}")

def runwebshell_debug(url):
    os.system(f"webshell {url} --debug")

def runwebshell_ssl(url, cert, key):
    os.system(f"webshell {url} --ssl-cert {cert} --ssl-key {key}")

def runwebshell_debug_ssl(url, cert, key):
    os.system(f"webshell {url} --debug --ssl-cert {cert} --ssl-key {key}")

#----------------------------kill-----------------------------

def killngrok():
    os.system("taskkill /f /im ngrok.exe")

def killflask():
    os.system("taskkill /f /im flask.exe")

def killwebshell():
    os.system("taskkill /f /im webshell.exe")

#----------------------line-api------------------------

class line_notify:
    def __init__(self, token):
        self.token = token

    def send(self, message):
        r = requests.post(f"https://notify-api.line.me/api/notify", headers={"Authorization": f"Bearer {self.token}"}, data={"message": message})
        if r.status_code == 200:
            return r.text
        else:
            raise line_api_exception(f"line notify error: {r.text}")

    def send_file(self, file):
        r = requests.post(f"https://notify-api.line.me/api/notify", headers={"Authorization": f"Bearer {self.token}"}, files={"imageFile": file})
        if r.status_code == 200:
            return r.text
        else:
            raise line_api_exception(f"LINE notify error: {r.status_code}")

#----------------------checker------------------------

def distrochecker(giftcode):
    r = requests.get(f'https://discordapp.com/api/v9/entitlements/gift-codes/{giftcode}?with_application=false&with_subscription_plan=true')
    if r == 200:
        return ('ok', giftcode)
    else:
        return ('error', giftcode)
