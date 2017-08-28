import datetime, time
import pyautogui
from pyautogui import typewrite, hotkey


pyautogui.FAILSAFE = False
print(
    datetime.datetime.fromtimestamp(
        int(time.time())
    ).strftime('%Y-%m-%d %H:%M:%S')
)
def convert_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(
        int(time.time())
    ).strftime('%Y-%m-%d %H:%M:%S')

i = 0
while True:
    time.sleep(60 * 5)
    print('Iteration: ', i, convert_to_datetime(time.time()))
    i += 1
    typewrite('a')
    # Sleep 60 seconds * 5 = 5 mins

