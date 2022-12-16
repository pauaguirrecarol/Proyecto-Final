#Imports for creating insert window
from kivy.uix.textinput import TextInput
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.metrics import dp
from kivymd.uix.button import MDFlatButton
from kivymd.uix.datatables import MDDataTable
from kivymd.uix.dialog import MDDialog
#from kivymd.uix.picker import MDDatePicker 
from kivy.uix.screenmanager import Screen
from kivymd.uix.filemanager import MDFileManager
from kivymd.toast import toast
from kivy.uix.image import Image
import sys
sys.path.insert(1,'/Users/paulaaguirrecarol/Desktop/Proyecto Final')
from funciones.funciones_todas import *



#Creating the class for the analizer window
class VentanaAnalizador(MDApp):
    def __init__(self, nombre, senal, fecha, **kwargs):
        #Creating a Screen
        self.senal = senal
        self.nombre = nombre
        self.fecha = fecha
        super().__init__(**kwargs)
   
    def build(self, *args):

        self.screen = Screen()
        self.theme_cls.primary_palette = "Gray"
        self.theme_cls.theme_style = "Light"

        #self.menu = MDDropdownMenu()

        #Creating BoxLayout
        box = BoxLayout(orientation='vertical')
        #Creating Title Label
        title = Label(text="Procesamiento de la señal", font_size=120, pos_hint={'center_x': 0.5, 'center_y': 0.99}, color = 'black', bold = True)
        box.add_widget(title)


        #Creating a GridLayout
        layout = GridLayout(cols = 2,orientation='lr-tb', spacing=dp(200), padding=dp(50), row_default_height=dp(30), row_force_default=True, size_hint=(1, .9), pos_hint={'center_x': 0.5, 'center_y': 0.9})
        #Creating a text input
        text_input = TextInput(font_size=50, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, text = self.nombre, disabled = True)
        

        layout.add_widget(text_input)
        #Creating a text input for the date
        text_input2 = TextInput(font_size=50, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        text_input2.text = self.fecha
        text_input2.disabled = True

        layout.add_widget(text_input2)

        box.add_widget(layout)
        #Creatin a secondary titlte
        title2 = Label(text="Señal", font_size=80, pos_hint={'center_x': 0.5, 'center_y': 1.2}, color = 'black',bold = True)
        box.add_widget(title2)
        #Creating an image box
        image_box = BoxLayout(orientation='vertical', size_hint=(dp(1), dp(1)), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        #Creating an image
        self.image = Image(source='imagenes/senal en blanco.png', pos_hint={'center_x': 0.5, 'center_y': 0.5})
        image_box.add_widget(self.image)
        box.add_widget(image_box)

        #adding a second grid layout
        layout2 = GridLayout(cols = 3,orientation='lr-tb', spacing=dp(10), padding=dp(20), row_default_height=dp(40), row_force_default=True)
        #Creating buttons
        button = Button(text="Señal Original", font_size=40, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, on_release = self.mostrarSenalOriginal)
        layout2.add_widget(button)
        button = Button(text="Espectro", font_size=40, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, on_release = self.mostrarEspectro)
        layout2.add_widget(button)
        button = Button(text="Eliminar Tendencias", font_size=40, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, on_release = self.eliminartendencias)
        layout2.add_widget(button)
        button = Button(text="FPM", font_size=40, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, on_release = self.FPM)
        layout2.add_widget(button)
        button = Button(text="Filtro FIR", font_size=40, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, on_release = self.FIR)
        layout2.add_widget(button)
        button = Button(text="Filtro IIR", font_size=40, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, on_release = self.IIR)
        layout2.add_widget(button)

        box.add_widget(layout2)
        self.screen.add_widget(box)


        return self.screen

    def mostrarSenalOriginal(self, obj):
        GraficarOriginalACQ (self.senal)
        self.image.source = 'imagenes/senalGeneradaACQ.jpg'


    def mostrarEspectro(self, obj):
        Espectro (self.senal)
        self.image.source = 'imagenes/espectro.jpg'


    def eliminartendencias(self, obj):
        TendenciaSenoidalDETREND (self.senal)
        self.image.source =  'imagenes/tendencia_senoidal_detrend.jpg'


    def FPM(self, obj):
        toast ("El filtro de orden ... posee una atenuación de ... y una SNR de ... dB.", duration=15)
        FPM (self.senal)
        self.image.source = 'imagenes/FPM.jpg'


    def FIR(self, obj):
        FIR (self.senal)
        self.image.source = 'imagenes/FIR.jpg'


    def IIR(self, obj):
        IIR (self.senal)
        self.image.source = 'imagenes/IIR.jpg'




if __name__ == "__main__":
    VentanaAnalizador(senal="/Users/paulaaguirrecarol/DesktopCopia de 2. ecg_sinf.acq",nombre="Olivia Carol",fecha="a").run()
