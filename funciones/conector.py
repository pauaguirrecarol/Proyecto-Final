import sqlite3
from sqlite3 import Error
#Define a conector to the sqlite3 database
def conector():
    try:
        conn = sqlite3.connect('DataBase.db')
        return conn
    except Error as e:
        print(e)
    return None

#Define a functionto insert data into the database
def insertar(conn, datos):
    sql = ''' INSERT INTO Pacientes(nombre,apellido,edad,sexo,fecha,diagnostico,señal)
              VALUES(?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, datos)
    conn.commit()
    return cur.lastrowid

def guardar_datos(nombre, apellido, edad, sexo, fecha, diagnostico, senal):
    conn = conector()
    datos = (nombre, apellido, edad, sexo, fecha, diagnostico, senal)
    insertar(conn, datos)
    conn.close()