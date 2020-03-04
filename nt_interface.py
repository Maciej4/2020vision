import time
from networktables import NetworkTables
import threading

table = None


class NTInterface:

    def __init__(self):
        global table
        cond = threading.Condition()
        notified = [False]

        def connection_listener(connected, info):
            print(info, '; Connected=%s' % connected)
            with cond:
                notified[0] = True

                cond.notify()

        # NetworkTables.initialize(server='10.9.72.2')
        # NetworkTables.initialize(server='192.168.86.254')
        NetworkTables.initialize(server='0.0.0.0')
        NetworkTables.addConnectionListener(connection_listener, immediateNotify=True)

        with cond:
            print("Waiting")
            if not notified[0]:
                cond.wait()
                # sys.exit()

        print("Connected!")

        table = NetworkTables.getTable('jetson')

        table.putNumber('heartbeat', -1)

        time.sleep(0.5)

    def put_num(self, key, value):
        global table
        table.putNumber(key, value)

    def put_str(self, key, string):
        global table
        table.putString(key, string)
