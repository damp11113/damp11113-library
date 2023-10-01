import codecs
import paramiko
import socket
import threading
import six
import logging
from functools import wraps

logger = logging.getLogger("PyserSSH")


system_banner = (
    f"\033[36mPyserSSH V2.0 \033[0m\n"
    f"\033[33m!!Warning!! This is Testing Version of PyserSSH \033[0m\n"
    #f"\033[35mUse '=' as spacer \033[0m\n"
    f"\033[35mUse Putty for best experience \033[0m\n"
    f"\033[33mDon't use MobaXterm because \033[31mBUG \033[0m\n"
)

def replace_enter_with_crlf(input_string):
    if '\n' in input_string:
        input_string = input_string.replace('\n', '\r\n')
    return input_string

class History():
    def __init__(self):
        self.command_history = []  # Store command history for each user
        self.currenthistory = 0  # Initialize currenthistory variable here

    def add(self, command):
        self.command_history.append(command)

    def getCommand(self, index):
        if 0 <= index < len(self.command_history):
            return self.command_history[-1 - index]
        return None

    def getAll(self):
        return self.command_history

    def modify(self, value):
        self.currenthistory += value

    def clear(self):
        self.command_history = []


class Server(paramiko.ServerInterface):
    def __init__(self, username="admin", password=""):
        self.event = threading.Event()
        self.current_user = None
        self.username = username
        self.password = password

    def check_channel_request(self, kind, chanid):
        if kind == 'session':
            return paramiko.OPEN_SUCCEEDED
        return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_auth_password(self, username, password):
        if (username == self.username) and (password == self.password):
            self.current_user = username  # Store the current user upon successful authentication
            return paramiko.AUTH_SUCCESSFUL

    def check_channel_pty_request(self, channel, term, width, height, pixelwidth, pixelheight, modes):
        return True

    def check_channel_shell_request(self, channel):
        return True

class SSHServer:
    def __init__(self, host, port, private_key_path, prompt=">", systemessage=True):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.private_key = paramiko.RSAKey(filename=private_key_path)
        self._event_handlers = {}
        self.prompt = prompt
        self.sysmess = systemessage
        self.disconnected = False

    def on_user(self, event_name):
        def decorator(func):
            @wraps(func)
            def wrapper(channel, *args, **kwargs):
                return func(channel, *args, **kwargs)
            self._event_handlers[event_name] = wrapper
            return wrapper
        return decorator

    def _handle_event(self, event_name, *args, **kwargs):
        handler = self._event_handlers.get(event_name)
        if handler:
            handler(*args, **kwargs)

    def run(self, username="admin", password=''):
        try:
            self.server.listen(100)
            logger.info("Start Listening for connections...")
            while True:
                client, addr = self.server.accept()
                bh_session = paramiko.Transport(client)
                bh_session.add_server_key(self.private_key)
                server = Server(username, password)
                bh_session.start_server(server=server)
                channel = bh_session.accept(20)
                if channel is None:
                    logger.warning("no channel")
                logger.info("user authenticated")
                self.disconnected = False
                if self.sysmess:
                    channel.sendall(replace_enter_with_crlf(system_banner))
                self._handle_event("connect", channel)
                peername = channel.getpeername()
                history = History()
                try:
                    channel.send(replace_enter_with_crlf(self.prompt + " ").encode('utf-8'))
                    while True:
                        self.expect(channel, history, peername)
                except KeyboardInterrupt:
                    channel.close()
                    bh_session.close()
                except Exception as e:
                    logger.error(e)
        except Exception as e:
            logger.error(e)

    def expect(self, chan, history, peername, echo=True):
        buffer = six.BytesIO()
        cursor_position = 0  # Variable to keep track of cursor position

        try:
            while True:
                byte = chan.recv(1)
                self._handle_event("ontype", chan, byte)

                if not byte or byte == b'\x04':
                    raise EOFError()
                elif byte == b'\t':
                    pass
                elif byte == b'\x7f':
                    if cursor_position > 0:
                        chan.sendall(b'\b \b')
                        buffer.truncate(buffer.tell() - 1)
                        cursor_position -= 1
                elif byte == b'\x1b' and chan.recv(1) == b'[':
                    arrow_key = chan.recv(1)
                    if arrow_key == b'A':
                        # Up arrow key, load previous command from history
                        history.currenthistory += 1
                        command_from_history = history.getCommand(history.currenthistory)
                        if command_from_history is not None:
                            chan.sendall(b'\r')
                            chan.sendall(replace_enter_with_crlf(self.prompt + command_from_history).encode('utf-8'))
                    elif arrow_key == b'B':
                        # Down arrow key, load next command from history
                        history.currenthistory -= 1
                        command_from_history = history.getCommand(history.currenthistory)
                        if command_from_history is not None:
                            chan.sendall(b'\r')
                            chan.sendall(replace_enter_with_crlf(self.prompt + command_from_history).encode('utf-8'))
                    elif arrow_key == b'C':
                        # Right arrow key, move cursor right if not at the end
                        if cursor_position < buffer.tell():
                            chan.sendall(b'\x1b[C')
                            cursor_position += 1
                    elif arrow_key == b'D':
                        # Left arrow key, move cursor left if not at the beginning
                        if cursor_position > 0:
                            chan.sendall(b'\x1b[D')
                            cursor_position -= 1
                elif byte in (b'\r', b'\n'):
                    break
                else:
                    buffer.write(byte)
                    cursor_position += 1
                    if echo:
                        chan.sendall(byte)

            if echo:
                chan.sendall(b'\r\n')

            history.add(codecs.decode(buffer.getvalue(), 'utf-8'))
            self._handle_event("command", chan, codecs.decode(buffer.getvalue(), 'utf-8'))
            try:
                chan.send(replace_enter_with_crlf(self.prompt + " ").encode('utf-8'))
            except:
                self.disconnected = True

        except Exception:
            raise
        finally:
            # Check if the user is disconnected
            if not byte:
                logger.info(f"{peername} is disconnected")
                self._handle_event("disconnected", peername)

