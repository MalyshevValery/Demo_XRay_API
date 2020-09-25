"""Listens for messages from starter and pushes them to callback"""
from multiprocessing import Queue
from .pill import PoisonPill


class AutoListener:
    """NO THREADING"""
    def __init__(self, in_conn: Queue, out_conn: Queue, callback):
        self.in_conn = in_conn
        self.out_conn = out_conn
        self.callback = callback

    def run(self):
        """Main loop"""
        while True:
            data = self.in_conn.get()
            try:
                if isinstance(data, PoisonPill):
                    break
                else:
                    print(data)
                    ret = self.callback(data)
            except Exception as e:
                ret = str(e)
            self.out_conn.put(ret)
