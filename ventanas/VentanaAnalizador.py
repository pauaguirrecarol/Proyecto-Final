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



#from funciones.Graficar import *
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
        text_input = TextInput(font_size=30, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, text = self.nombre, disabled = True)
        

        layout.add_widget(text_input)
        #Creating a text input for the date
        text_input2 = TextInput(font_size=30, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        text_input2.text = self.fecha
        text_input2.disabled = True

        layout.add_widget(text_input2)

        box.add_widget(layout)
        #Creatin a secondary titlte
        title2 = Label(text="Señal", font_size=70, pos_hint={'center_x': 0.5, 'center_y': 0.99}, color = 'black',bold = True)
        box.add_widget(title2)
        #Creating an image box
        image_box = BoxLayout(orientation='vertical', size_hint=(dp(1), dp(1)), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        #Creating an image
        self.image = Image(source='imagenes/senal.jpg', pos_hint={'center_x': 0.5, 'center_y': 0.5})
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
        #self.buttonEliminar = MDFlatButton(text="Eliminar Tendencias", font_size=40, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, id = 'eliminar', on_release = self.menu.open())
        layout2.add_widget(button)
        button = Button(text="FPB", font_size=40, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, on_release = self.FPB)
        layout2.add_widget(button)
        button = Button(text="Filtro FIR", font_size=40, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, on_release = self.FIR)
        layout2.add_widget(button)
        button = Button(text="Filtro IIR", font_size=40, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, on_release = self.IIR)
        layout2.add_widget(button)

        box.add_widget(layout2)
        self.screen.add_widget(box)

        #-Menu desplegable--------------------------------
        # menu_items = [
        #     {
        #         "text": f"Item {i}",
        #         "right_text": f"R+{i}",
        #         "right_icon": "apple-keyboard-command",
        #         "left_icon": "git",
        #         "viewclass": "Item",
        #         "height": dp(54),
        #         "on_release": lambda x=f"Item {i}": self.menu_callback(x),
        #     } for i in range(5)
        # ]
        # self.menu = MDDropdownMenu(
        #     caller=self.ids.buttonEliminar,
        #     items=menu_items,
        #     width_mult=4,
        # )


        return self.screen

    def mostrarSenalOriginal(self, obj):
        toast("Señal Original")
        #GraficarOriginal(self.senal)
        #self.image.source = 'imagenes/senal.jpg'


    def mostrarEspectro(self, obj):
        toast("Espectro")
    def eliminartendencias(self, obj):
        toast("Eliminar Tendencias")
    def FPB(self, obj):
        toast("FPB")
    def FIR(self, obj):
        toast("Filtro FIR")
    def IIR(self, obj):
        toast("Filtro IIR")


    #--------------------------------------------
    #def menu_callback(self, text_item):
    #    toast(text_item)
    #--------------------------------------------


#---------------------------Listas desplegables--------------------------------
# class RightContentCls(IRightBodyTouch, MDBoxLayout):
#     icon = StringProperty()
#     text = StringProperty()


# class Item(OneLineAvatarIconListItem):
#     left_icon = StringProperty()
#     right_icon = StringProperty()
#     right_text = StringProperty()





if __name__ == "__main__":
    VentanaAnalizador(senal="a",nombre="Olivia Carol").run()
