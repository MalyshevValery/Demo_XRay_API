import os
import socket
import subprocess
import threading


class AutoStarter:
    def __init__(self, timeout, call, socket_name, keyword='DIE'):
        self.proc = None
        self.timer = None

        self.timeout = float(timeout)
        self.call = call
        self.socket_name = socket_name
        self.keyword = keyword
        self.lock = threading.Lock();

        if os.path.exists(self.socket_name):
            os.remove(self.socket_name)

    def __start(self):
        if os.path.exists(self.socket_name) or self.proc is not None:
            raise Exception('False start')
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(self.socket_name)
        s.listen(1)
        self.proc = subprocess.Popen(self.call)
        self.conn, addr = s.accept()

    def stop(self):
        if self.proc is None:
            return
        self.conn.send(self.keyword.encode('UTF-8'))
        self.proc.wait()
        self.conn.close()
        self.proc = None
        self.timer = None
        self.conn = None
        os.remove(self.socket_name)
        return 0

    def send_recv(self, input_):
        self.lock.acquire()
        if self.proc is None:
            self.__start()
        elif self.timer is not None:
            self.timer.cancel()
        self.lock.release()
        self.conn.send((input_ + '\n').encode('UTF-8'))
        recv = self.conn.recv(1024).decode('UTF-8')
        self.timer = threading.Timer(self.timeout, self.stop)
        self.timer.start()
        return recv