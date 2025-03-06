import tensorflow as tf
import numpy as np
from ripser import ripser
from persim import plot_diagrams, bottleneck
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# Define a simple model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Client update function
def client_update(model, dataset):
    model.compile(optimizer='sgd', loss='binary_crossentropy')
    model.fit(dataset, epochs=1, verbose=0)
    return model.get_weights()

# Server aggregation function with noise injection
def server_aggregate_with_noise(weights_list, add_noise=False, noise_scale=0.1):
    aggregated_weights = [np.mean(weights, axis=0) for weights in zip(*weights_list)]

    if add_noise:
        # Add noise to aggregated weights
        noisy_weights = []
        for weight in aggregated_weights:
            noise = np.random.normal(0, noise_scale, weight.shape)
            noisy_weights.append(weight + noise)
        return noisy_weights
    else:
        return aggregated_weights

# Compute persistence diagram
def compute_persistence_diagram(weights):
    flattened_weights = np.concatenate([w.flatten() for w in weights])
    diagrams = ripser(flattened_weights.reshape(-1, 1))['dgms']
    return diagrams[0]  # Return 1-dimensional features

# Detect attack using persistence diagrams
def detect_attack(current_diagram, previous_diagram, threshold):
    if previous_diagram is None:
        return False
    distance = bottleneck(current_diagram, previous_diagram)
    return distance > threshold

# Simulate federated learning
num_clients = 10
num_rounds = 20
noise_scale = 0.1 # Increased noise for more noticeable effect
noisy_rounds = [5, 10, 15,20]  # Specify the rounds where you want to add noise
attack_threshold =0.05   # Adjust this threshold for attack detection sensitivity

# Initialize global model
global_model = create_model()
previous_diagram = None


for round in range(num_rounds):
    print(f"Round {round + 1}/{num_rounds}")

    # Simulate client updates
    client_weights = []
    for _ in range(num_clients):
        X = np.random.rand(1000, 5)
        y = np.random.randint(0, 2, 1000)
        dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)

        client_model = create_model()
        client_model.set_weights(global_model.get_weights())
        client_weights.append(client_update(client_model, dataset))

    # Server aggregation with noise in specific rounds
    add_noise = (round + 1) in noisy_rounds
    aggregated_weights = server_aggregate_with_noise(client_weights, add_noise=add_noise, noise_scale=noise_scale)
    global_model.set_weights(aggregated_weights)

    # Compute persistence diagram
    current_diagram = compute_persistence_diagram(aggregated_weights)

    # Detect attack
    is_attack_detected = detect_attack(current_diagram, previous_diagram, attack_threshold)

    #Plot persistence diagram
    plt.figure(figsize=(10, 5))
    plot_diagrams([current_diagram], show=False)
    plt.title(f"Persistence Diagram - Round {round + 1}" +
              (" (Noisy)" if add_noise else "") +
              (" - ATTACK DETECTED" if is_attack_detected else ""))
    plt.savefig(f"persistence_diagram_round_{round + 1}.png")
    #plt.close()

    if add_noise:
        print(f"Noise added in round {round + 1}")
    if is_attack_detected:
        print(f"ATTACK DETECTED in round {round + 1}")

    previous_diagram = current_diagram

print("Federated learning completed.")
