#Imports for creating insert window
from kivy.uix.textinput import TextInput
from kivymd.uix.pickers import MDDatePicker, MDTimePicker
from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.metrics import dp
from kivymd.uix.button import MDFlatButton
from kivymd.uix.datatables import MDDataTable
from kivymd.uix.button import MDFillRoundFlatButton
from kivy.uix.screenmanager import Screen
from kivymd.uix.filemanager import MDFileManager
from kivymd.toast import toast
from kivy.uix.image import Image
import os
#import sys
#sys.path.insert(1,'/Users/mateo/Desktop/Proyecto-Final')
from VentanaAnalizador import VentanaAnalizador

#creating the class for the insert window
class VentanaIngreso(MDApp):
    def build(self, *args, **kwargs):
        #Creating a Screen
        screen = Screen()
        self.theme_cls.primary_palette = "Gray"
        self.theme_cls.theme_style = "Light"

        #Creating BoxLayout
        box = BoxLayout(orientation='vertical')
        #Creating Title Label
        title = Label(text="Ingresar datos del paciente", font_size=100, pos_hint={'center_x': 0.5, 'center_y': 0.99}, color = 'black',bold = True, size_hint=(1, .5))


        #Creating a GridLayout
        layout = GridLayout(cols = 2,orientation='lr-tb', spacing=dp(10), padding=dp(10), row_default_height=dp(50), row_force_default=True)
        box.add_widget(title,)
        #Creating a label
        label = Label(text="Nombre:", font_size=50, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, color = 'black')
        #Adding the label to the GridLayout
        layout.add_widget(label)
        #Creating a text input
        self.nombre = TextInput(multiline=False, font_size=65, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        #Adding the text input to the GridLayout
        layout.add_widget(self.nombre)

        #Creating a label
        label2 = Label(text="Apellido:", font_size=50, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, color = 'black')
        #Adding the label to the GridLayout
        layout.add_widget(label2)
        #Creating a text input
        self.apellido = TextInput(multiline=False, font_size=65, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        #Adding the text input to the GridLayout
        layout.add_widget(self.apellido)

        #Creating a label
        label3 = Label(text="Edad:", font_size=50, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, color = 'black')
        #Adding the label to the GridLayout
        layout.add_widget(label3)
        #Creating a text input
        text_input3 = TextInput(multiline=False, font_size=65, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        #Adding the text input to the GridLayout
        layout.add_widget(text_input3)

        #Creating a label
        label4 = Label(text="Sexo:", font_size=50, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, color = 'black')
        #Adding the label to the GridLayout
        layout.add_widget(label4)
        #Creating a text input
        text_input4 = TextInput(multiline=False, font_size=65, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        #Adding the text input to the GridLayout
        layout.add_widget(text_input4)

        #Creating a label
        label5 = Label(text="Diagnóstico:", font_size=50, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, color = 'black')
        #Adding the label to the GridLayout
        layout.add_widget(label5)
        #Creating a text input
        text_input5 = TextInput(multiline=False, font_size=65, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        #Adding the text input to the GridLayout
        layout.add_widget(text_input5)

        #-------------------------                
        #Creating a label
        label6 = MDFillRoundFlatButton(text="Seleccionar fecha de ingreso", font_size=50, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5},md_bg_color = 'gray', on_press=self.show_date_picker)
        #Adding the label to the GridLayout
        layout.add_widget(label6)
        #Creating a text input
        self.textofecha = TextInput(multiline=False, font_size=65, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, disabled=True)
        #Adding the text input to the GridLayout
        layout.add_widget(self.textofecha)
        #Creating a button
        #button = Button(text="Seleccionar", font_size=40, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        #Adding the button to the GridLayout
        #layout.add_widget(button)
        #-------------------------

        #Creating a file chooser
        self.file_manager  = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path,
        )

        #creating a button to open the file chooser
        button = MDFillRoundFlatButton(text="Subir Archivo:", font_size=50, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, md_bg_color = 'gray',on_release=self.file_manager_open)
        #Adding the button to the GridLayout
        layout.add_widget(button)
        #Creating a text input
        self.textosenal = TextInput(multiline=False, font_size=40, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, disabled=True)
        #Adding the text input to the GridLayout
        layout.add_widget(self.textosenal)

        #Creating a button
        button = Button(text="Atrás", font_size=50, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, background_normal = "",background_color = 'black',on_release=self.salir)
        #Adding the button to the GridLayout
        layout.add_widget(button)
        #Creating a button
        button = Button(text="Guardar", font_size=50, size_hint=(.2, .2), pos_hint={'center_x': 0.5, 'center_y': 0.5}, background_normal = "",background_color = 'black',on_release=self.guardar)
        #Adding the button to the GridLayout
        layout.add_widget(button)






        box.add_widget(layout)

        screen.add_widget(box, index=3)

    
        return screen

    def on_save(self, instance, value, date_range):
        '''
        Events called when the "OK" dialog box button is clicked.

        :type instance: <kivymd.uix.picker.MDDatePicker object>;

        :param value: selected date;
        :type value: <class 'datetime.date'>;

        :param date_range: list of 'datetime.date' objects in the selected range;
        :type date_range: <class 'list'>;
        '''
        self.fecha = str(value).format("%d/%m/%Y")
        self.textofecha.text = self.fecha
        print(instance, value, date_range)

    def on_cancel(self, instance, value):
        '''Events called when the "CANCEL" dialog box button is clicked.'''

    def show_date_picker(self, *args):
       date_dialog = MDDatePicker()
       date_dialog.bind(on_save=self.on_save, on_cancel=self.on_cancel)
       date_dialog.open()

    def file_manager_open(self, *args):
        self.file_manager.show(os.path.expanduser("~"))  # output manager to the screen
        self.manager_open = True

    def select_path(self, path: str):
        '''
        It will be called when you click on the file name
        or the catalog selection button.

        :param path: path to the selected directory or file;
        '''


        self.senal = path
        print(self.senal)
        self.textosenal.text = self.senal
        self.exit_manager()
        toast(path)

    def exit_manager(self, *args):
        '''Called when the user reaches the root of the directory tree.'''

        self.manager_open = False
        self.file_manager.close()

    def events(self, instance, keyboard, keycode, text, modifiers):
        '''Called when buttons are pressed on the mobile device.'''

        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()
        return True

    def salir(self, *args):
        self.stop()
        print("Volver")
    
    def guardar(self, *args):
        if self.nombre.text == "" or self.apellido.text == "" or self.textosenal.text == "" or self.textofecha.text == "":
            toast("Faltan datos")
            return
        self.stop()
        self.root.clear_widgets()
        nombre = self.nombre.text + " " + self.apellido.text 
        print(nombre)
        VentanaAnalizador(nombre=nombre,senal=self.senal,fecha=self.fecha).run()
        print("Guardar")
        VentanaIngreso().run()


if __name__ == '__main__':
    VentanaIngreso().run()
