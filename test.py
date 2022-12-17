import sys
sys.path.insert(1,'/Users/paulaaguirrecarol/Desktop/Proyecto Final')
from funciones.conector import conector, insertar

#Creating a functionn to save the data
def guardar_datos(nombre, apellido, edad, sexo, fecha, diagnostico, senal):
    conn = conector()
    datos = (nombre, apellido, edad, sexo, fecha, diagnostico, senal)
    insertar(conn, datos)
    conn.close()


if __name__ == '__main__':
    guardar_datos('Paula', 'Aguirre', 18, 'F', '2020-12-12', 'Sano', 'Señal')