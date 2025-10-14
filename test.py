import time

from global_utils import WindowsRouser

if __name__ == '__main__':
    rouser = WindowsRouser()
    rouser.start()
    time.sleep(1000)
