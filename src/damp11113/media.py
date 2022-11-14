import vlc
import pafy, pafy2
import ffmpeg
import cv2
import tqdm
import numpy as np
import qrcode
import barcode
from barcode.writer import ImageWriter
import os
from PIL import Image
from . import sort_files, get_size_unit
from .randoms import rannum
from .file import sizefolder3, removefile, allfiles
import yt_dlp as youtube_dl2
from pyzbar import pyzbar
import colorsys

class vlc_player:
    def __init__(self):
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

class youtube_stream2:
    def __init__(self, url):
        self.stream = pafy2.new(url)
        self.ydl = youtube_dl2.YoutubeDL()
        self.url = url

    def video_stream(self, resolution=None):
        if resolution is None:
            best = self.stream.getbestvideo()
        else:
            best = self.stream.getbestvideo().resolution(resolution)
        return best.url

    def audio_stream(self):
        best = self.stream.getbestaudio()
        return best.url

    def best_stream(self, preftype="mp4"):
        best = self.stream.getbest(preftype=preftype)
        return best.url

    def get_streams(self):
        return self.ydl.extract_info(self.url, download=False)['formats']

    def get_id(self):
        return self.ydl.extract_info(self.url, download=False)['id']

    def get_title(self):
        return self.ydl.extract_info(self.url, download=False)['title']

    def get_dec(self):
        return self.ydl.extract_info(self.url, download=False)['description']


class ffmpeg_stream:
    def __init__(self) -> None:
        pass

    def load(self, file_path):
        self.stream = ffmpeg.input(file_path)
        return f"Loaded {file_path}"

    def write(self, file_path, acodec='mp4', ac=2, hz='44100', bitrate='320'):
        ffmpeg.run(self.stream.output(file_path, acodec=acodec, ac=ac, ar=hz, **{'b:a': f'{bitrate}k'}))
        return f"Writing {file_path}"

    def streaming(self, url, acodec='mp4', ac=2, hz='44100', bitrate='320'):
        ffmpeg.run(self.stream.output(url, acodec=acodec, ac=ac, ar=hz, **{'b:a': f'{bitrate}k'}))
        return f"Streaming {url}"


def clip2frames(clip_path, frame_path, currentframe=1, filetype='png'):
    try:
        clip = cv2.VideoCapture(clip_path)
        length = int(clip.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = tqdm.tqdm(total=length, unit='frame')
        progress.set_description(f'set output to {frame_path}')
        if not os.path.exists(frame_path):
            os.mkdir(frame_path)
            progress.set_description(f'create output folder {frame_path}')
        progress.set_description(f'converting... ')
        while True:
            size = get_size_unit(sizefolder3(frame_path))
            ret, frame = clip.read()
            PILframe = CV22PIL(frame)
            PILframe.save(f'{frame_path}/{str(currentframe)}' + f'.{filetype}')
            progress.set_description(f'converting... | filetype .{filetype} | converted {currentframe}/{length} | file {currentframe}.{filetype} | used {size}')
            currentframe += 1
            progress.update(1)
            if currentframe == length:
                progress.set_description(f'converted {currentframe} frame | used {size} MB')
                progress.close()
                break
    except Exception as e:
        progress = tqdm.tqdm(total=0)
        progress.set_description(f'error: {e}')
        progress.close()

def im2ascii(image, width=None, height=None, new_width=None, chars=None, pixelss=25):
    try:
        try:
            img = Image.open(image)
            img_flag = True
        except:
            print(image, "Unable to find image ")

        if width is None:
            width = img.size[0]
        if height is None:
            height = img.size[1]
        if new_width is None:
            new_width = width
        aspect_ratio = int(height)/int(width)
        new_height = aspect_ratio * new_width * 0.55
        img = img.resize((new_width, int(new_height)))

        img = img.convert('L')

        if chars is None:
            chars = ["@", "J", "D", "%", "*", "P", "+", "Y", "$", ",", "."]

        pixels = img.getdata()
        new_pixels = [chars[pixel//pixelss] for pixel in pixels]
        new_pixels = ''.join(new_pixels)
        new_pixels_count = len(new_pixels)
        ascii_image = [new_pixels[index:index + new_width] for index in range(0, new_pixels_count, new_width)]
        ascii_image = "\n".join(ascii_image)
        return ascii_image

    except Exception as e:
        raise e

def im2pixel(image, i_size, output):
    img = Image.open(image)
    small_img = img.resize(i_size, Image.BILINEAR)
    res = small_img.resize(img.size, Image.NEAREST)
    res.save(output)

def qrcodegen(text, showimg=False, save_path='./', filename='qrcode', filetype='png', version=1, box_size=10, border=5, fill_color="black", back_color="white", error_correction=qrcode.constants.ERROR_CORRECT_L, fit=True):
    qr = qrcode.QRCode(
        version=version,
        error_correction=error_correction,
        box_size=box_size,
        border=border,
    )
    qr.add_data(text)
    qr.make(fit=fit)
    img = qr.make_image(fill_color=fill_color, back_color=back_color)
    if showimg:
        img.show()
    else:
        img.save(f'{save_path}{filename}.{filetype}')


def barcodegen(number, type='ean13', showimg=False, save_path='./', filename='barcode', filetype='png', writer=ImageWriter()):
    barcode_img = barcode.get(type, number, writer=writer)
    if showimg:
        img = Image.open(barcode_img.save(f'{save_path}{filename}.{filetype}'))
        img.show()
        removefile(barcode_img.save(f'{save_path}{filename}.{filetype}'))
    else:
        barcode_img.save(f'{save_path}{filename}.{filetype}')

def imseq2clip(imseq, path, videoname='video.mp4', fps=30):
    progress = tqdm.tqdm()
    progress.set_description(f'please wait...')
    simseq = sort_files(allfiles(imseq), reverse=False)
    img = []
    for i in simseq:
        i = path+i
        img.append(i)
    progress.set_description(f'converting...')
    progress.total = len(img)
    progress.unit = 'frame'
    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path+videoname, cv2_fourcc, fps)
    for i in range(len(img)):
        progress.set_description(f'converting... | frame {i}/{len(img)}')
        frame = cv2.imread(img[i])
        video.write(frame)
        progress.update(1)

    video.release()
    progress.set_description(f'converted')

def readbqrcode(image):
    image = Image.open(image)
    qr_code = pyzbar.decode(image)[0]
    data = qr_code.data.decode("utf-8")
    type = qr_code.type
    return (data, type)

def PIL2CV2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def CV22PIL(cv2_array):
    return Image.fromarray(cv2.cvtColor(cv2_array, cv2.COLOR_BGR2RGB))

def image_high_pass(image, opath):
    img = cv2.imread(image)
    hpf = img - cv2.GaussianBlur(img, (21, 21), 3)+127
    cv2.imwrite(opath, hpf)

def image_levels(image, minv=0, maxv=255, gamma=1.0):
    class Level(object):
        def __init__(self, minv, maxv, gamma):
            self.minv= minv/255.0
            self.maxv= maxv/255.0
            self._interval= self.maxv - self.minv
            self._invgamma= 1.0/gamma

        def new_level(self, value):
            if value <= self.minv: return 0.0
            if value >= self.maxv: return 1.0
            return ((value - self.minv)/self._interval)**self._invgamma

        def convert_and_level(self, band_values):
            h, s, v= colorsys.rgb_to_hsv(*(i/255.0 for i in band_values))
            new_v= self.new_level(v)
            return tuple(int(255*i)
                    for i
                    in colorsys.hsv_to_rgb(h, s, new_v))
    """Level the brightness of image (a PIL.Image instance)
    All values ≤ minv will become 0
    All values ≥ maxv will become 255
    gamma controls the curve for all values between minv and maxv"""

    if image.mode != "RGB":
        raise ValueError("this works with RGB images only")

    new_image= image.copy()

    leveller= Level(minv, maxv, gamma)
    levelled_data= [
        leveller.convert_and_level(data)
        for data in image.getdata()]
    new_image.putdata(levelled_data)
    return new_image

def image2stripes(image, opath, radius=10, minv=0, maxv=255, gamma=1.0):
    background = Image.open(image)
    img = PIL2CV2(background)
    hpf = img - cv2.GaussianBlur(img, ((radius * 10) +1, (radius * 10) +1), 1)+(9*20)
    output = image_levels(CV22PIL(hpf), minv=minv, maxv=maxv, gamma=gamma)
    output.save(opath)

def ranpix(opath, size=(512, 512)):
    im = Image.new("RGB", size=size)
    width, height = im.size
    for x in range(width):
        for y in range(height):
            L = rannum(0, 255)
            R = rannum(0, 255)
            G = rannum(0, 255)
            B = rannum(0, 255)
            LRGB = (L, R, G, B)
            im.putpixel((x, y), LRGB)
    im.save(opath)

def PIL2DPG(pil_image):
    return CV22DPG(PIL2CV2(pil_image))

def CV22DPG(cv2_array):
    data = np.flip(cv2_array, 2)
    data = data.ravel()
    data = np.asfarray(data, dtype='f')
    return np.true_divide(data, 255.0)