import numpy as np
import serial
import time

# Configuración de la conexión Bluetooth (ajusta los parámetros según sea necesario)
bluetooth_port = 'COM3'  # Cambia esto al puerto correcto
baud_rate = 9600
ser = serial.Serial(bluetooth_port, baud_rate)
time.sleep(2)  # Espera a que se establezca la conexión

# Cargar la tabla Q desde el archivo
q_table = np.load('q_table.npy')

# Definir las acciones y sus comandos correspondientes para el coche Arduino
acciones = ['Adelante', 'Izquierda', 'Derecha']
comandos = {'Adelante': 'A', 'Izquierda': 'L', 'Derecha': 'R'}

# Estado inicial
estado_actual = 0

# Función para determinar la siguiente acción basada en la tabla Q
def determinar_accion(estado):
    return np.argmax(q_table[estado, :])

# Enviar las acciones al coche Arduino
while estado_actual != len(q_table) - 1:  # Mientras no estemos en el estado final
    accion_idx = determinar_accion(estado_actual)
    accion = acciones[accion_idx]
    comando = comandos[accion]

    # Enviar el comando al coche Arduino
    ser.write(comando.encode())
    print(f'Enviando acción: {accion}')

    # Esperar un tiempo para que el coche ejecute la acción
    time.sleep(2)  # Ajusta este tiempo según sea necesario

    # Actualizar el estado actual (este es un ejemplo básico; actualiza según tu lógica)
    if accion == 'Adelante':
        estado_actual += 1  # Supón que avanzar lleva al siguiente estado
    elif accion == 'Izquierda' or accion == 'Derecha':
        estado_actual = (estado_actual + 1) % len(q_table)  # Simula un giro y avanza

# Cerrar la conexión Bluetooth
ser.close()
print('Se completó el envío de acciones.')
