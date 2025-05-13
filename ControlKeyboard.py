#Imports Here
import pyautogui
from typing import List

class ControlKeyboard:
    def __init__(self):
        # Model Loading Part here
        pass

    def control_keyboard(self, key) -> List[tuple]:
        pyautogui.press(key)
        pass