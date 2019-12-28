
import sys
import os
import time
 
class Logger(object):
    def __init__(self, filename="Default.log"):
        path = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(path,filename)
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
 
    def flush(self):
        pass

if __name__ =='__main__':

    logfileName = 'log' + str(int(time.time()))+'.txt'
    sys.stdout = Logger(logfileName)

    print('test')
    print('hhh')
    time.sleep(10)