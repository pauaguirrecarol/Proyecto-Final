#Imports for creating main window
from kivymd.app import MDApp

from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.gridlayout import GridLayout
from kivy.metrics import dp


#creating the class for the main window
class VentanaPrincipal(MDApp):
    def build(self):
        #Creating a Screen
        screen = Screen()
        #Setting the background color to gray
        self.theme_cls.primary_palette = "Gray"
        self.theme_cls.theme_style = "Light"

        #Creating Title Label
        title = Label(text="Plataforma de análisis de señales de ECG", font_size=100, pos_hint={'center_x': 0.5, 'center_y': 0.5}, color = 'black', bold = True)
        #screen.add_widget(title)
        #Adding an image
        image = Image(source='imagenes/1200px-Normal_ECG_2.svg.png',pos_hint={'center_x': 0.5, 'center_y': 0.7})
        image.size_hint = (dp(2), dp(2))

        #Creating a GridLayout
        box = BoxLayout(orientation='vertical')
        box.add_widget(title)
        box.add_widget(image)
        layout = GridLayout(cols = 2,orientation='lr-tb', spacing=dp(50), padding=dp(10), row_default_height=dp(50), row_force_default=True)
        #Creating a button
        button = Button(text="Iniciar", font_size=50, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, on_press=self.iniciar)
        #Adding the button to the GridLayout
        #Creating a button
        button2 = Button(text="Salir", font_size=50, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, on_press=self.salir)
        #Adding the button to the GridLayout
        layout.add_widget(button2)
        layout.add_widget(button)
        box.add_widget(layout)
        #Adding the GridLayout to the Screen
        screen.add_widget(box, index=3)



        return screen


    def salir(self,obj):
        exit()

    def iniciar(self,obj):
        print("Iniciar")


if __name__ == '__main__':
    VentanaPrincipal().run()