import os

try:
    os.environ["damp11113_load_all_module"]
except:
    os.environ["damp11113_load_all_module"] = "YES"

if os.environ["damp11113_load_all_module"] == "YES":
    from . import *
    from .file import *
    from .network import *
    from .info import *
    from .randoms import *
    from .processbar import *
    from .media import *
    from .cmd import *
    from .convert import *
    from .imageps import *
    from .utils import *
    from .minecraft import *
    from .plusmata import *
    from .imageps import *
    from .DSP import *
    from .OPFONMW.dearpygui_animate import *
    from .OPFONMW.pyFMRDSEncoder.Encoder import *
    from .OPFONMW.pyFMRDSEncoder.DataGenerator import *
    from .logic import *




ip = 'https://cdn.damp11113dev.tk'
__version__ = '2023.10.1.22.0.0' # 2022/12/7 | 22 file (no __init__.py) | --- function |

try:
    os.environ["damp11113_check_update"]
except:
    os.environ["damp11113_check_update"] = "YES"

if os.environ["damp11113_check_update"] == "YES":
    from pygments import console
    import requests
    print(console.colorize("yellow", "library check update..."))
    try:
        response = requests.get(f"{ip}/file/text/damp11113libver.txt")
        if response.status_code == 200:
            if response.text == __version__:
                print(f'{console.colorize("green", "no update available")}')
                print(f'{console.colorize("green", f"library version current: {__version__}")}')
            else:
                print(console.colorize("yellow", "update available"))
                print(f'{console.colorize("green", f"library version current: {__version__}")}')
                print(f'{console.colorize("green", f"new: {response.text}")}')
        else:
            print(f'{console.colorize("red", f"check update failed. please try again (error {response.status_code})")}')
            print(f'{console.colorize("yellow", f"library version current: {__version__}")}')

    except:
        print(console.colorize("red", "check update failed. please try again"), f'{__version__}')
        print(f'{console.colorize("yellow", f"library version current: {__version__}")}')