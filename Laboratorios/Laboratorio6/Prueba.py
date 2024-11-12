import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Función de entrenamiento para el entorno FrozenLake-v1
def train(episodes, map_name="4x4", is_slippery=False):
    # Inicializa el entorno de FrozenLake-v1 de Gymnasium
    env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery)
    # Crea la tabla Q con todos los valores en 0
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # Define los parámetros del algoritmo
    learning_rate = 0.1  # Tasa de Aprendizaje
    discount_factor = 0.95  # Factor de Descuento para recompensas futuras
    epsilon = 1.0  # Probabilidad inicial de escoger una acción aleatoria
    epsilon_decay = 0.001  # Decaimiento de epsilon
    rng = np.random.default_rng()  # Generador de números aleatorios
    rewards_per_episode = np.zeros(episodes)  # Lista para almacenar las recompensas por episodio

    # Bucle principal de entrenamiento
    for i in range(episodes):
        # Reinicia el entorno cada 1000 episodios, alternando entre modos con y sin renderizado
        if (i + 1) % 1000 == 0:
            env.close()
            env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery, render_mode='human')
        else:
            env.close()
            env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery)

        # Reinicia el entorno y obtiene el estado inicial
        state = env.reset()[0]
        terminated = False
        truncated = False

        while (not terminated and not truncated):
            # Selecciona una acción
            if rng.random() < epsilon:
                # Exploración
                action = env.action_space.sample()  # Acción aleatoria
            else:
                # Explotación
                action = np.argmax(q_table[state, :])  # Acción según la tabla Q

            # Realiza la acción y obtiene el nuevo estado y la recompensa
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Actualiza la tabla Q
            q_table[state, action] = q_table[state, action] + learning_rate * (
                reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action])

            # Actualiza el estado
            state = next_state

        # Decaimiento de epsilon para disminuir la exploración
        epsilon = max(epsilon - epsilon_decay, 0)

        # Registrar la recompensa obtenida en el episodio
        rewards_per_episode[i] = reward

        # Imprime el progreso cada 100 episodios
        if (i + 1) % 100 == 0:
            print(f'Episodio {i + 1} - Recompensa: {rewards_per_episode[i]}')

    # Cierra el entorno
    env.close()

    # Impresión de la tabla Q
    print(f"Tabla Q: {q_table}")

    # Calcula y muestra la suma de recompensas obtenidas
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])
    plt.plot(sum_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Suma de Recompensas')
    plt.title('Suma de Recompensas por Episodios')
    plt.show()

    
# Entrena el agente
if __name__ == '__main__':
    train(5000)
