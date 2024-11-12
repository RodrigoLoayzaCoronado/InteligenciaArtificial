import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import serial
import time
#PROYECTO DE APRENDIZAJE POR REFUERZO IMPLEMENTADO A ARDUINO 
#  
# Establecer conexión con Arduino mediante Bluetooth (ajusta el puerto y la velocidad según sea necesario)
bluetooth = serial.Serial('COM8', 38400)  # Cambia 'COM8' por el puerto
time.sleep(2)  # Espera para asegurar la conexión

# Entorno personalizado simple de 5x5 con orientación del coche y obstáculos
class SimpleGridEnv(gym.Env):
    def __init__(self):
        self.grid_size = 5
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Discrete(self.grid_size * self.grid_size),  # 25 posibles estados de posición
            gym.spaces.Discrete(4)  # 4 posibles orientaciones: 0=Arriba, 1=Derecha, 2=Abajo, 3=Izquierda
        ))
        self.action_space = gym.spaces.Discrete(4)  # 4 posibles acciones: avanzar, girar izq, girar der, frenar
        self.state = (0, 0)  # Estado inicial: (posición, orientación)
        self.destination = 24  # El destino es la última celda

        # Definir obstáculos como una lista de posiciones (índices de la cuadrícula)
        self.obstacles = {5, 3, 7, 13, 18}  # Celdas que son obstáculos

    def reset(self):
        self.state = (0, 0)  # Reinicia el estado inicial (posición, orientación)
        return self.state, {}

    def step(self, action):
        position, orientation = self.state
        row, col = divmod(position, self.grid_size)

        new_row, new_col = row, col
        if action == 0:  # Avanzar
            if orientation == 0 and row > 0:  # Arriba
                new_row -= 1
            elif orientation == 1 and col < self.grid_size - 1:  # Derecha
                new_col += 1
            elif orientation == 2 and row < self.grid_size - 1:  # Abajo
                new_row += 1
            elif orientation == 3 and col > 0:  # Izquierda
                new_col -= 1

        elif action == 1:  # Girar a la izquierda
            orientation = (orientation - 1) % 4

        elif action == 2:  # Girar a la derecha
            orientation = (orientation + 1) % 4

        elif action == 3:  # Frenar
            pass  # No cambia ni la posición ni la orientación

        new_position = new_row * self.grid_size + new_col

        # Si la nueva posición es un obstáculo, el coche no se mueve y se aplica una penalización
        if new_position in self.obstacles:
            new_position = position  # Mantener la posición actual

        reward = -1  # Penalización por cada movimiento
        done = False

        if new_position == self.destination:
            reward = 20  # Recompensa por llegar al destino
            done = True

        self.state = (new_position, orientation)
        return self.state, reward, done, False, {}

    def render(self, mode='human'):
        grid = np.zeros((self.grid_size, self.grid_size))
        position, _ = self.state
        row, col = divmod(position, self.grid_size)
        grid[row, col] = 1  # Marcar la posición del coche

        # Marcar los obstáculos en la cuadrícula
        for obs in self.obstacles:
            obs_row, obs_col = divmod(obs, self.grid_size)
            grid[obs_row, obs_col] = -1  # Marcar obstáculos

        plt.imshow(grid, cmap='Blues', vmin=-1, vmax=1)
        plt.show()

# Función de entrenamiento
def train(episodes):
    env = SimpleGridEnv()
    q_table = np.zeros([env.observation_space.spaces[0].n, env.observation_space.spaces[1].n, env.action_space.n])

    learning_rate = 0.1
    discount_factor = 0.95
    epsilon = 1.0
    epsilon_decay = 0.001
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    # Bucle de entrenamiento
    for i in range(episodes):
        state = env.reset()[0]
        done = False

        while not done:
            if rng.random() < epsilon:  # Explorar: elegir acción aleatoria
                action = env.action_space.sample()
            else:  # Explotar: elegir la mejor acción según la tabla Q
                action = np.argmax(q_table[state[0], state[1], :])

            next_state, reward, done, _, _ = env.step(action)

            # Actualizar el valor Q
            q_table[state[0], state[1], action] += learning_rate * (
                reward + discount_factor * np.max(q_table[next_state[0], next_state[1], :]) - q_table[state[0], state[1], action])

            state = next_state

        # Reducir el valor de epsilon para disminuir la exploración con el tiempo
        epsilon = max(epsilon - epsilon_decay, 0)
        rewards_per_episode[i] = reward

        # Mostrar progreso cada 100 episodios
        if (i + 1) % 100 == 0:
            print(f'Episodio {i + 1} - Recompensa: {rewards_per_episode[i]}')

    # Guardar la tabla Q entrenada en un archivo
    np.save('q_table.npy', q_table)
    print(f"Tabla Q guardada: {q_table}")

    # Graficar la suma de recompensas
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])
    plt.plot(sum_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Suma de Recompensas')
    plt.title('Suma de Recompensas por Episodios')
    plt.show()

    # Demostración del agente entrenado
    state = env.reset()[0]
    done = False
    rewards = 0

    while not done:
        action = np.argmax(q_table[state[0], state[1], :])
        state, reward, done, _, _ = env.step(action)
        rewards += reward
        env.render()

        # Enviar la acción a Arduino mediante Bluetooth
        bluetooth.write(f"{action}\n".encode())  # Envía la acción como cadena de texto
        print(f"Acción a realizar: {action}")
        time.sleep(2)  # Espeara dos segundos entre acciones para que Arduino pueda procesarlas

        if done:
            break

    print(f"Recompensa total del agente entrenado: {rewards}")

if __name__ == '__main__':
    train(1000)  # Entrena el agente por 1000 episodios
