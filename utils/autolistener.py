import socket


class AutoListener:
    """NO THREADING"""
    def __init__(self, socket_name, on_recv, keyword='DIE'):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(socket_name)
        while True:
            data = s.recv(1024)
            try:
                if data != bytes(keyword, encoding='UTF-8'):
                    if data == b'':
                        ret = 'EMPTY'
                    else:
                        data = data.decode('UTF-8')
                        print(data)
                        on_recv(data)
                        ret = 'SUCCESS'
                else:
                    break
            except Exception as e:
                ret = str(e)
            s.send(ret.encode('UTF-8'))
        s.close()
