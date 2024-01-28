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

import codecs
import os
import pickle
import time
import paramiko
import socket
import threading
import six
import logging
from functools import wraps
from .info import pyofetch
from .utils import TextFormatter

sftpclient = ["WinSCP", "Xplore"]

#paramiko.sftp_file.SFTPFile.MAX_REQUEST_SIZE = pow(2, 22)

logger = logging.getLogger("PyserSSH")
logger.disabled = True

version = "3.5"

system_banner = (
    f"\033[36mPyserSSH V{version} \033[0m\n"
    #"\033[33m!!Warning!! This is Testing Version of PyserSSH \033[0m\n"
    "\033[35mUse Putty and WinSCP (SFTP) for best experience \033[0m"
)

print(system_banner)

def replace_enter_with_crlf(input_string):
    if '\n' in input_string:
        input_string = input_string.replace('\n', '\r\n')
    return input_string

def Send(channel, string, ln=True):
    if ln:
        channel.send(replace_enter_with_crlf(string + "\n"))
    else:
        channel.send(replace_enter_with_crlf(string))

def wait_input(channel, prompt="", defaultvalue=None, cursor_scroll=False):
    channel.send(replace_enter_with_crlf(prompt))

    buffer = bytearray()
    cursor_position = 0

    try:
        while True:
            byte = channel.recv(1)

            if not byte or byte == b'\x04':
                raise EOFError()

            elif byte == b'\t':
                pass

            elif byte == b'\x7f':  # Backspace
                if cursor_position > 0:
                    # Move cursor back, erase character, move cursor back again
                    channel.sendall(b'\b \b')
                    buffer = buffer[:cursor_position - 1] + buffer[cursor_position:]
                    cursor_position -= 1

            elif byte == b'\x1b' and channel.recv(1) == b'[':  # Arrow keys
                arrow_key = channel.recv(1)
                if cursor_scroll:
                    if arrow_key == b'C':  # Right arrow key
                        if cursor_position < len(buffer):
                            channel.sendall(b'\x1b[C')
                            cursor_position += 1
                    elif arrow_key == b'D':  # Left arrow key
                        if cursor_position > 0:
                            channel.sendall(b'\x1b[D')
                            cursor_position -= 1

            elif byte in (b'\r', b'\n'):  # Enter key
                break

            else:  # Regular character
                buffer = buffer[:cursor_position] + byte + buffer[cursor_position:]
                cursor_position += 1
                channel.sendall(byte)

        channel.sendall(b'\r\n')

        output = buffer.decode('utf-8')

        # Return default value if specified and no input given
        if defaultvalue is not None and not output.strip():
            return defaultvalue
        else:
            return output

    except Exception:
        raise

def _systemcommand(channel, command, user):
    if command == "info":
        Send(channel, "Please wait...", ln=False)
        pyf = pyofetch().info(f"{TextFormatter.format_text('PyserSSH Version', color='yellow')}: {TextFormatter.format_text(version, color='cyan')}")
        Send(channel, "              \r", ln=False)
        for i in pyf:
            Send(channel, i)
    elif command == "whoami":
        Send(channel, user)
    elif command == "exit":
        channel.close()
    else:
        return False

class AccountManager:
    def __init__(self):
        self.accounts = {}

    def validate_credentials(self, username, password):
        if username in self.accounts and self.accounts[username]["password"] == password:
            return True
        return False

    def get_permissions(self, username):
        if username in self.accounts:
            return self.accounts[username]["permissions"]
        return []

    def set_prompt(self, username, prompt=">"):
        if username in self.accounts:
            self.accounts[username]["prompt"] = prompt

    def get_prompt(self, username):
        if username in self.accounts and "prompt" in self.accounts[username]:
            return self.accounts[username]["prompt"]
        return ">"  # Default prompt if not set for the user

    def add_account(self, username, password, permissions):
        self.accounts[username] = {"password": password, "permissions": permissions}

    def change_password(self, username, new_password):
        if username in self.accounts:
            self.accounts[username]["password"] = new_password

    def set_permissions(self, username, new_permissions):
        if username in self.accounts:
            self.accounts[username]["permissions"] = new_permissions

    def save_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.accounts, file)

    def load_from_file(self, filename):
        try:
            with open(filename, 'rb') as file:
                self.accounts = pickle.load(file)
        except FileNotFoundError:
            print("File not found. No accounts loaded.")
        except Exception as e:
            print(f"An error occurred: {e}. No accounts loaded.")

    def set_user_sftp_allow(self, username, allow=True):
        if username in self.accounts:
            self.accounts[username]["sftp_allow"] = allow

    def get_user_sftp_allow(self, username):
        if username in self.accounts and "sftp_allow" in self.accounts[username]:
            return self.accounts[username]["sftp_allow"]
        return True

    def set_user_sftp_readonly(self, username, readonly=False):
        if username in self.accounts:
            self.accounts[username]["sftp_readonly"] = readonly

    def get_user_sftp_readonly(self, username):
        if username in self.accounts and "sftp_readonly" in self.accounts[username]:
            return self.accounts[username]["sftp_readonly"]
        return False

    def set_user_sftp_path(self, username, path="/"):
        if username in self.accounts:
            if path == "/":
                self.accounts[username]["sftp_path"] = ""
            else:
                self.accounts[username]["sftp_path"] = path

    def get_user_sftp_path(self, username):
        if username in self.accounts and "sftp_path" in self.accounts[username]:
            return self.accounts[username]["sftp_path"]
        return ""


class Server(paramiko.ServerInterface):
    def __init__(self, accounts):
        self.event = threading.Event()
        self.current_user = None
        self.accounts = accounts

    def check_channel_request(self, kind, chanid):
        if kind == 'session':
            return paramiko.OPEN_SUCCEEDED
        return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_auth_password(self, username, password):
        if self.accounts.validate_credentials(username, password):
            self.current_user = username  # Store the current user upon successful authentication
            return paramiko.AUTH_SUCCESSFUL

    def check_channel_pty_request(self, channel, term, width, height, pixelwidth, pixelheight, modes):
        return True

    def check_channel_shell_request(self, channel):
        return True

    def check_channel_x11_request(self, channel, single_connection, auth_protocol, auth_cookie, screen_number):
        return True

class SSHSFTPHandle(paramiko.SFTPHandle):
    def stat(self):
        try:
            return paramiko.SFTPAttributes.from_stat(os.fstat(self.readfile.fileno()))
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)

    def chattr(self, attr):
        # python doesn't have equivalents to fchown or fchmod, so we have to
        # use the stored filename
        try:
            paramiko.SFTPServer.set_file_attr(self.filename, attr)
            return paramiko.SFTP_OK
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)

class SSHSFTPServer(paramiko.SFTPServerInterface):
    ROOT = None
    ACCOUNT = None
    CLIENTHANDELES = None

    def _realpath(self, path):
        return self.ROOT + self.canonicalize(path)

    def list_folder(self, path):
        path = self._realpath(path)
        try:
            out = []
            flist = os.listdir(path)
            for fname in flist:
                attr = paramiko.SFTPAttributes.from_stat(os.stat(os.path.join(path, fname)))
                attr.filename = fname
                out.append(attr)
            return out
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)

    def stat(self, path):
        path = self._realpath(path)
        try:
            return paramiko.SFTPAttributes.from_stat(os.stat(path))
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)

    def lstat(self, path):
        path = self._realpath(path)
        try:
            return paramiko.SFTPAttributes.from_stat(os.lstat(path))
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)

    def open(self, path, flags, attr):
        path = self._realpath(path)
        try:
            binary_flag = getattr(os, 'O_BINARY', 0)
            flags |= binary_flag
            mode = getattr(attr, 'st_mode', None)
            if mode is not None:
                fd = os.open(path, flags, mode)
            else:
                # os.open() defaults to 0777 which is
                # an odd default mode for files
                fd = os.open(path, flags, 0o666)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)
        if (flags & os.O_CREAT) and (attr is not None):
            attr._flags &= ~attr.FLAG_PERMISSIONS
            paramiko.SFTPServer.set_file_attr(path, attr)
        if flags & os.O_WRONLY:
            if flags & os.O_APPEND:
                fstr = 'ab'
            else:
                fstr = 'wb'
        elif flags & os.O_RDWR:
            if flags & os.O_APPEND:
                fstr = 'a+b'
            else:
                fstr = 'r+b'
        else:
            # O_RDONLY (== 0)
            fstr = 'rb'
        try:
            f = os.fdopen(fd, fstr)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)
        fobj = SSHSFTPHandle(flags)
        fobj.filename = path
        fobj.readfile = f
        fobj.writefile = f
        return fobj

    def remove(self, path):
        path = self._realpath(path)
        try:
            os.remove(path)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)
        return paramiko.SFTP_OK

    def rename(self, oldpath, newpath):
        oldpath = self._realpath(oldpath)
        newpath = self._realpath(newpath)
        try:
            os.rename(oldpath, newpath)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)
        return paramiko.SFTP_OK

    def mkdir(self, path, attr):
        path = self._realpath(path)
        try:
            os.mkdir(path)
            if attr is not None:
                paramiko.SFTPServer.set_file_attr(path, attr)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)
        return paramiko.SFTP_OK

    def rmdir(self, path):
        path = self._realpath(path)
        try:
            os.rmdir(path)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)
        return paramiko.SFTP_OK

    def chattr(self, path, attr):
        path = self._realpath(path)
        try:
            paramiko.SFTPServer.set_file_attr(path, attr)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)
        return paramiko.SFTP_OK

    def symlink(self, target_path, path):
        path = self._realpath(path)
        if (len(target_path) > 0) and (target_path[0] == '/'):
            # absolute symlink
            target_path = os.path.join(self.ROOT, target_path[1:])
            if target_path[:2] == '//':
                # bug in os.path.join
                target_path = target_path[1:]
        else:
            # compute relative to path
            abspath = os.path.join(os.path.dirname(path), target_path)
            if abspath[:len(self.ROOT)] != self.ROOT:
                # this symlink isn't going to work anyway -- just break it immediately
                target_path = '<error>'
        try:
            os.symlink(target_path, path)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)
        return paramiko.SFTP_OK

    def readlink(self, path):
        path = self._realpath(path)
        try:
            symlink = os.readlink(path)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)

        if os.path.isabs(symlink):
            if symlink[:len(self.ROOT)] == self.ROOT:
                symlink = symlink[len(self.ROOT):]
                if (len(symlink) == 0) or (symlink[0] != '/'):
                    symlink = '/' + symlink
            else:
                symlink = '<error>'
        return symlink

class SSHServer:
    def __init__(self, accounts, system_message=True, timeout=0, disable_scroll_with_arrow=True, sftp=True, sftproot=os.getcwd(), system_commands=False):
        """
         A simple SSH server
        """
        self._event_handlers = {}
        self.sysmess = system_message
        self.client_handlers = {}  # Dictionary to store event handlers for each client
        self.current_users = {}  # Dictionary to store current_user for each connected client
        self.accounts = accounts
        self.timeout = timeout
        self.disable_scroll_with_arrow = disable_scroll_with_arrow
        self.sftproot = sftproot
        self.sftpena = sftp
        self.enasyscom = system_commands

        self.system_banner = system_banner

        if self.enasyscom:
            print("\033[33m!!Warning!! System commands is enable! \033[0m")

    def on_user(self, event_name):
        def decorator(func):
            @wraps(func)
            def wrapper(channel, *args, **kwargs):
                # Ignore the third argument
                filtered_args = args[:2] + args[3:]
                return func(channel, *filtered_args, **kwargs)
            self._event_handlers[event_name] = wrapper
            return wrapper
        return decorator

    def _handle_event(self, event_name, *args, **kwargs):
        handler = self._event_handlers.get(event_name)
        if handler:
            handler(*args, **kwargs)

    def handle_client(self, client, addr):
        bh_session = paramiko.Transport(client)
        bh_session.add_server_key(self.private_key)

        if self.sftpena:
            SSHSFTPServer.ROOT = self.sftproot
            SSHSFTPServer.ACCOUNT = self.accounts
            SSHSFTPServer.CLIENTHANDELES = self.client_handlers
            bh_session.set_subsystem_handler('sftp', paramiko.SFTPServer, SSHSFTPServer)

        server = Server(self.accounts)
        bh_session.start_server(server=server)

        bh_session.default_window_size = 2147483647
        bh_session.packetizer.REKEY_BYTES = pow(2, 40)
        bh_session.packetizer.REKEY_PACKETS = pow(2, 40)

        logger.info(bh_session.remote_version)

        channel = bh_session.accept()


        if self.timeout != 0:
            channel.settimeout(self.timeout)

        if channel is None:
            logger.warning("no channel")

        logger.info("user authenticated")
        client_address = channel.getpeername()  # Get client's address to identify the user
        if client_address not in self.client_handlers:
            # Create a new event handler for this client if it doesn't exist
            self.client_handlers[client_address] = {
                "event_handlers": {},
                "current_user": None,
                "channel": channel,  # Associate the channel with the client handler,
                "last_activity_time": None,
                "connecttype": None,
                "last_login_time": None
            }
        client_handler = self.client_handlers[client_address]
        client_handler["current_user"] = server.current_user
        client_handler["channel"] = channel  # Update the channel attribute for the client handler
        client_handler["last_activity_time"] = time.time()
        client_handler["last_login_time"] = time.time()

        peername = channel.getpeername()

        #if not any(bh_session.remote_version.split("-")[2].startswith(prefix) for prefix in sftpclient):
        if not channel.out_window_size == bh_session.default_window_size:
            if self.sysmess:
                channel.sendall(replace_enter_with_crlf(self.system_banner))
                channel.sendall(replace_enter_with_crlf("\n"))

            self._handle_event("connect", channel, self.client_handlers[channel.getpeername()]["current_user"])

            client_handler["connecttype"] = "ssh"
            try:
                channel.send(replace_enter_with_crlf(self.accounts.get_prompt(self.client_handlers[channel.getpeername()]["current_user"]) + " ").encode('utf-8'))
                while True:
                    self.expect(channel, peername)
            except KeyboardInterrupt:
                channel.close()
                bh_session.close()
            except Exception as e:
                logger.error(e)
            finally:
                client.close()
        else:
            if self.sftpena:
                if self.accounts.get_user_sftp_allow(self.client_handlers[channel.getpeername()]["current_user"]):
                    client_handler["connecttype"] = "sftp"
                    self._handle_event("connectsftp", channel, self.client_handlers[channel.getpeername()]["current_user"])
                else:
                    client.close()
            else:
                client.close()

    def stop_server(self):
        logger.info("Stopping the server...")
        try:
            for client_handler in self.client_handlers.values():
                channel = client_handler.get("channel")
                if channel:
                    channel.close()
            self.server.close()
            logger.info("Server stopped.")
        except Exception as e:
            logger.error(f"Error occurred while stopping the server: {e}")

    def _start_listening_thread(self):
        try:
            self.server.listen(10)
            logger.info("Start Listening for connections...")
            while True:
                client, addr = self.server.accept()
                client_thread = threading.Thread(target=self.handle_client, args=(client, addr))
                client_thread.start()

        except Exception as e:
            logger.error(e)

    def run(self, private_key_path, host="0.0.0.0", port=2222):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
        self.server.bind((host, port))
        self.private_key = paramiko.RSAKey(filename=private_key_path)

        client_thread = threading.Thread(target=self._start_listening_thread)
        client_thread.start()

    def expect(self, chan, peername, echo=True):
        buffer = bytearray()
        cursor_position = 0

        try:
            while True:
                byte = chan.recv(1)
                self._handle_event("ontype", chan, byte, self.client_handlers[chan.getpeername()]["current_user"])

                if self.timeout != 0:
                    self.client_handlers[chan.getpeername()]["last_activity_time"] = time.time()

                if not byte or byte == b'\x04':
                    raise EOFError()
                elif byte == b'\t':
                    pass
                elif byte == b'\x7f':
                    if cursor_position > 0:
                        buffer = buffer[:cursor_position - 1] + buffer[cursor_position:]
                        cursor_position -= 1
                        chan.sendall(b'\b \b')
                elif byte == b'\x1b' and chan.recv(1) == b'[':
                    arrow_key = chan.recv(1)
                    if not self.disable_scroll_with_arrow:
                        if arrow_key == b'C':
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
                    buffer = buffer[:cursor_position] + byte + buffer[cursor_position:]
                    cursor_position += 1
                    if echo:
                        chan.sendall(byte)

            if echo:
                chan.sendall(b'\r\n')

            command = buffer.decode('utf-8')
            currentuser = self.client_handlers[chan.getpeername()]["current_user"]

            if self.enasyscom:
                _systemcommand(chan, command, currentuser)

            self._handle_event("command", chan, command, currentuser)
            try:
                chan.send(replace_enter_with_crlf(self.accounts.get_prompt(currentuser) + " ").encode('utf-8'))
            except:
                logger.error("Send error")

        except Exception as e:
            logger.error(str(e))
        finally:
            if not byte:
                logger.info(f"{peername} is disconnected")
                self._handle_event("disconnected", peername, self.client_handlers[peername]["current_user"])

    def kickbyusername(self, username, reason=None):
        for peername, client_handler in list(self.client_handlers.items()):
            if client_handler["current_user"] == username:
                channel = client_handler.get("channel")
                if reason is None:
                    if channel:
                        channel.close()
                    del self.client_handlers[peername]
                    logger.info(f"User '{username}' has been kicked.")
                else:
                    if channel:
                        Send(channel, f"You have been disconnected for {reason}")
                        channel.close()
                    del self.client_handlers[peername]
                    logger.info(f"User '{username}' has been kicked by reason {reason}.")

    def kickbypeername(self, peername, reason=None):
        client_handler = self.client_handlers.get(peername)
        if client_handler:
            channel = client_handler.get("channel")
            if reason is None:
                if channel:
                    channel.close()
                del self.client_handlers[peername]
                logger.info(f"peername '{peername}' has been kicked.")
            else:
                if channel:
                    Send(channel, f"You have been disconnected for {reason}")
                    channel.close()
                del self.client_handlers[peername]
                logger.info(f"peername '{peername}' has been kicked by reason {reason}.")

    def kickall(self, reason=None):
        for peername, client_handler in self.client_handlers.items():
            channel = client_handler.get("channel")
            if reason is None:
                if channel:
                    channel.close()
            else:
                if channel:
                    Send(channel, f"You have been disconnected for {reason}")
                    channel.close()

        if reason is None:
            self.client_handlers.clear()
            logger.info("All users have been kicked.")
        else:
            logger.info(f"All users have been kicked by reason {reason}.")

    def broadcast(self, message):
        for client_handler in self.client_handlers.values():
            channel = client_handler.get("channel")
            if channel:
                try:
                    # Send the message to the client
                    Send(channel, message)
                except Exception as e:
                    logger.error(f"Error occurred while broadcasting message: {e}")

    def sendto(self, username, message):
        for client_handler in self.client_handlers.values():
            if client_handler.get("current_user") == username:
                channel = client_handler.get("channel")
                if channel:
                    try:
                        # Send the message to the specific client
                        Send(channel, message)
                    except Exception as e:
                        logger.error(f"Error occurred while sending message to {username}: {e}")
                    break
        else:
            logger.warning(f"User '{username}' not found.")