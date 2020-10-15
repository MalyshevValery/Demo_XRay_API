"""Process starter class"""
from multiprocessing import Queue, Process, Lock
from threading import Thread, Timer
from queue import Empty

from xray_processing.main_utils import main as routine

from .pill import PoisonPill



class AutoStarter:
    """Class for automatic starting and shutdown of another process as well as
    for blocking communication"""

    def __init__(self, timeout, call):
        self.proc = None
        self.timer = None

        self.timeout = float(timeout)
        self.call = call
        self.lock = Lock()
        self.par_q = Queue()
        self.child_q = Queue()
        self.starter_queue = Queue()

        self.starter_thread = Thread(target=self.__starter_routine, daemon=True)
        self.starter_thread.start()

    def start(self):
        self.starter_queue.put(1)

    def __starter_routine(self):
        while True:
            self.starter_queue.get()
            if self.proc is None:
                self.__start()

    def __start(self):
        if self.proc is not None:
            raise Exception('False start')
        print('Start')
        self.proc = Process(target=routine, args=(self.par_q, self.child_q))
        self.proc.start()

    def stop(self):
        """Stops the AutoStarter"""
        if self.proc is None:
            return
        self.par_q.put(PoisonPill())
        self.proc.join()
        self.proc = None
        self.timer = None
        return 0

    def send_recv(self, input_):
        """Sends message and wait for reply"""
        self.lock.acquire()
        if self.proc is None:
            self.start()
        elif self.timer is not None:
            self.timer.cancel()
        self.par_q.put(input_)
        recv = self.child_q.get()
        self.timer = Timer(self.timeout, self.stop)
        self.timer.start()
        self.lock.release()
        return recv
