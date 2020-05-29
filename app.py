import kivy
kivy.require('1.10.1')
from kivy.uix.screenmanager import ScreenManager, Screen, NoTransition
from kivy.base import ExceptionHandler, ExceptionManager
from kivy.app import App
from kivy.properties import ObjectProperty, StringProperty
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.factory import Factory


import os
import cv2

app = Builder.load_file("movement_tracking.kv")


class MainWindow(Screen):
    video = ObjectProperty(None)
    stats = ObjectProperty(None)

    def run(self):
        os.system("python movement_tracking.py")



class Analytics(Screen):
    pass

class BackButton(Image):
    pass 

class Manager(ScreenManager):
    main_screen = ObjectProperty(None)
    analytics = ObjectProperty(None)


class BrowseFiles(Screen):
    file_path = StringProperty("No file choosen")

    def load_video(self, selected):
        try:
            if selected[0].lower().endswith('.mp4'):
                self.file_path = str(selected[0]).replace('\\', '/')
                self.load_video(self.file_path)
                os.system("python movement_tracking.py --video " + '"' + self.file_path.replace('\\', '/') + '"')
            else:
                popup = MsgPopup(e)
                popup.open()
        except Exception as e:
            popup = MsgPopup(e)
            popup.open()
        

    def update_file_list_entry(self, file_chooser, file_list_entry, *args):
        file_list_entry.children[0].color = (0.0, 0.0, 0.0, 1.0)  # File Names
        file_list_entry.children[1].color = (0.0, 0.0, 0.0, 1.0)


class MsgPopup(Popup):
    def __init__(self, msg):
        super().__init__()
        self.ids.message_label.text = "Niste odabrali ni jednu datoteku ili datoteka nije MP4 formata"

    def dismiss_popup(self):
        self.dismiss()


class Movement_Tracking_App(App):

    def build(self):
        title = "Movement Tracking App"
        m = Manager(transition=NoTransition())
        return m


if __name__ == '__main__':
    Movement_Tracking_App().run()
