import os
import json
from datetime import datetime
from neuron import Neuron
from neuron_ensemble_monitor import NeuronEnsembleMonitor
from state_mgr.neuron_state_manager import save_neuron_state, load_neuron_state
from state_mgr.monitor_state_manager import save_monitor_state, load_monitor_state
from mappings.data_loader import load_pixel
import numpy as np
import importlib

# Use config.py for first loop, config_training.py for subsequent loops
def load_config(granular=True, training=False):
    module_name = 'config_training' if training else 'config'
    config_module = importlib.import_module(module_name)
    config_vars = {
        'NUM_NEURONS': config_module.NUM_NEURONS,
        'NUM_CLUSTERS': config_module.NUM_CLUSTERS,
        'COLOR_LABELS': config_module.COLOR_LABELS,
        'COLOR_RGBS': config_module.COLOR_RGBS,
        'TRAINING_LOOPS': config_module.TRAINING_LOOPS,
        'CLUSTER_SPECIALTIES': getattr(config_module, 'CLUSTER_SPECIALTIES', [2, 6, 10]),
    }
    if granular:
        config_vars.update({
            'ACTIVATION_THRESHOLD': config_module.ACTIVATION_THRESHOLD,
            'WEIGHT_RANGE': config_module.WEIGHT_RANGE,
            'CONFIDENCE_INIT': config_module.CONFIDENCE_INIT,
            'CONFIDENCE_UPDATE_CORRECT': config_module.CONFIDENCE_UPDATE_CORRECT,
            'CONFIDENCE_UPDATE_INCORRECT': config_module.CONFIDENCE_UPDATE_INCORRECT,
            'MESSAGE_NOISE_PROB': config_module.MESSAGE_NOISE_PROB,
            'MESSAGE_FLIP_PROB': config_module.MESSAGE_FLIP_PROB,
            'MESSAGE_CONFIDENCE_THRESHOLD': config_module.MESSAGE_CONFIDENCE_THRESHOLD,
            'SPECIALTY_LEARNING_RATE': config_module.SPECIALTY_LEARNING_RATE,
            'SPECIALTY_REPULSION_STRENGTH': config_module.SPECIALTY_REPULSION_STRENGTH,
            'SPECIALTY_REPULSION_DISTANCE': config_module.SPECIALTY_REPULSION_DISTANCE,
            'SPECIALTY_CLAMP_MIN': config_module.SPECIALTY_CLAMP_MIN,
            'SPECIALTY_CLAMP_MAX': config_module.SPECIALTY_CLAMP_MAX,
            'WEIGHT_NOISE_RANGE': config_module.WEIGHT_NOISE_RANGE,
            'LEARNING_NOISE_RANGE': config_module.LEARNING_NOISE_RANGE,
            'RANDOM_FLIP_PROB': config_module.RANDOM_FLIP_PROB,
        })
    globals().update(config_vars)

LOGS_DIR = "logs"

def get_log_filename():
    """Generate log filename based on current date."""
    today = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    return os.path.join(LOGS_DIR, f"{today}.json")

def append_run_log(summary, log_filename):
    """Append experiment summary to daily log file."""
    try:
        # Try to load existing logs
        if os.path.exists(log_filename):
            try:
                with open(log_filename, "r") as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Corrupted log file {log_filename}, creating new one")
                logs = []
        else:
            logs = []
        
        # Add timestamp to summary
        summary["timestamp"] = datetime.now().isoformat()
        logs.append(summary)
        
        # Write logs back to file
        with open(log_filename, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"Log appended to {log_filename}")
    except Exception as e:
        print(f"Error saving log: {e}")
        # Try to save as backup
        backup_filename = log_filename.replace('.json', '_backup.json')
        try:
            with open(backup_filename, "w") as f:
                json.dump([summary], f, indent=2)
            print(f"Saved as backup: {backup_filename}")
        except Exception as backup_error:
            print(f"Failed to save backup: {backup_error}")

def train_green_detection():
    """Main training function for green detection experiment."""
    print("Starting green detection experiment...")
    # Load and update run_count at the start
    try:
        with open("run_count.txt", "r") as f:
            run_count = int(f.read().strip())
    except Exception:
        run_count = 0
    run_count += 1
    with open("run_count.txt", "w") as f:
        f.write(str(run_count))

    # Use config values for number of neurons
    num_neurons = globals().get('NUM_NEURONS', 9)

    print("[Step 1] Loading or initializing neurons and monitor...")
    neurons = load_neuron_state("network_state.pkl")
    if neurons is None:
        print("  Creating new neural network...")
        neurons = []
        for i in range(num_neurons):
            cluster_idx = i % globals().get('NUM_CLUSTERS', 3)
            specialties = globals().get('CLUSTER_SPECIALTIES', [2, 6, 10])
            specialty = specialties[cluster_idx] + np.random.uniform(-0.5, 0.5)
            neuron = Neuron(position=(i,), specialty=specialty, cluster=cluster_idx)
            neurons.append(neuron)
        for i in range(len(neurons)):
            if i > 0:
                neurons[i].add_neighbor(neurons[i-1])
            if i < len(neurons) - 1:
                neurons[i].add_neighbor(neurons[i+1])
    else:
        print(f"  Loaded {len(neurons)} neurons from state.")

    monitor = load_monitor_state("monitor_state.pkl")
    # Always reinitialize monitor to reference current neurons
    print("  Creating new monitor (always references current neurons)...")
    monitor = NeuronEnsembleMonitor(neurons)

    print("[Step 2] Loading and processing input pixel...")
    import random
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    pixel = load_pixel((r, g, b))
    print(f"  Input pixel: {pixel}")

    # Randomize goal
    color_labels = globals().get('COLOR_LABELS', ["red", "green", "blue"])
    goal_options = [f"detect_{label}" for label in color_labels]
    goal = random.choice(goal_options)
    print(f"[Step 2b] Goal for this run: {goal}")

    # Only clear round state after logging activity

    print("[Step 4] Setting goal for all neurons...")
    for neuron in neurons:
        neuron.receive_goal(goal)

    print("[Step 5] Calculating input signal from selected color component...")
    # Build color_map from config
    color_map = {f"detect_{label}": idx for idx, label in enumerate(color_labels)}
    color_idx = color_map.get(goal, 0)
    # Use selected color channel as signal
    input_signal = pixel[color_idx] / 255.0
    print(f"  Input signal strength: {input_signal}")

    print("[Step 6] Evaluating activation for all clusters (with weight noise)...")
    # Assign neurons to clusters
    num_clusters = globals().get('NUM_CLUSTERS', 3)
    cluster_size = max(1, len(neurons) // num_clusters)
    clusters = []
    for i in range(num_clusters):
        cluster_neurons = neurons[i*cluster_size:(i+1)*cluster_size]
        from cluster import Cluster
        clusters.append(Cluster(cluster_neurons, cluster_id=f"C{i}"))
    # Create controller
    from cluster_controller import ClusterController
    controller = ClusterController(clusters, controller_id="TrainController")
    # Add weight noise to each neuron in cluster
    for cluster in clusters:
        for neuron in cluster.neurons:
            noise = random.uniform(-0.05, 0.05)
            neuron.weight = min(1.0, max(0.0, neuron.weight + noise))
        cluster.aggregate_signal(input_signal)
        for idx, neuron in enumerate(cluster.neurons):
            line = f"    Neuron {neuron.position[0]}: specialty={neuron.specialty:.4f}, weight={neuron.weight:.4f}, active={'True' if neuron.active else 'False'}, confidence={neuron.confidence:.4f}"
            if hasattr(neuron, 'last_guess') and neuron.active:
                line += f", guess={getattr(neuron, 'last_guess', None)}, correct?={'yes' if getattr(neuron, 'last_guess', None) == getattr(neuron, 'goal', None) else 'no'}"
            print(line)
    # --- Equalize confidence after cluster activation ---
    controller.equalize_confidence()

    # --- Equalize specialty after cluster activation ---
    controller.equalize_specialty()

    print("[Step 7] Neurons sending messages to neighbors (with message noise)...")
    for idx, neuron in enumerate(neurons):
        # Message noise: randomly drop or alter messages
        original_send_message = neuron.send_message
        def noisy_send_message():
            if neuron.active and neuron.has_fired_this_round:
                for neighbor in neuron.neighbors:
                    # 10% chance to drop message
                    if random.random() < 0.1:
                        continue
                    # 10% chance to flip message type
                    specialty_diff = abs(neuron.specialty - neighbor.specialty)
                    if random.random() < 0.1:
                        msg_type = 'excite' if specialty_diff >= 2.0 else 'inhibit'
                    else:
                        msg_type = 'excite' if specialty_diff < 2.0 else 'inhibit'
                    neighbor.receive_message({'type': msg_type, 'from': neuron.position})
        neuron.send_message = noisy_send_message
        neuron.send_message()
        neuron.send_message = original_send_message
        print(f"    Neuron {idx}: sent messages, buffer={neuron.message_buffer}")

    print("[Step 8] Neuron learning step (with learning noise)...")
    for idx, neuron in enumerate(neurons):
        old_specialty = neuron.specialty
        learning_noise = random.uniform(-0.1, 0.1)
        if neuron.active:
            neuron.learn_specialty(input_signal)
            neuron.specialty += learning_noise
        else:
            learning_rate = 0.05
            neuron.specialty += learning_rate * (input_signal - neuron.specialty) + learning_noise
        print(f"    Neuron {idx}: specialty before={old_specialty:.4f}, after={neuron.specialty:.4f}, iteration={neuron.iteration}, learning_noise={learning_noise:+.4f}")

    print("[Step 9] Logging activity and generating summary...")
    monitor.log_activity(input_label=goal.replace("detect_", ""))
    summary = monitor.summarize()
    monitor.print_human_summary()

    # Visualize specialty values over time (save to CSV)
    specialties = [neuron.specialty for neuron in neurons]
    csv_header = "run," + ",".join([f"specialty_{i}" for i in range(num_neurons)]) + "\n"
    csv_line = f"{run_count}," + ",".join([f"{s:.4f}" for s in specialties]) + "\n"
    csv_path = "specialties_over_time.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write(csv_header)
    with open(csv_path, "a") as f:
        f.write(csv_line)
    print(f"[Step 9b] Specialty values saved to {csv_path}")

    print("[Step 10] Clearing previous round state...")
    for neuron in neurons:
        neuron.clear_round()

    print("[Step 11] Saving state for next run...")
    # Optionally reset state every N runs (here, every 10 runs)
    reset_interval = 10
    try:
        with open("run_count.txt", "r") as f:
            run_count = int(f.read().strip())
    except Exception:
        run_count = 0
    run_count += 1
    with open("run_count.txt", "w") as f:
        f.write(str(run_count))
    if run_count % reset_interval == 0:
        print(f"  Resetting state after {reset_interval} runs!")
        if os.path.exists("network_state.pkl"): os.remove("network_state.pkl")
        if os.path.exists("monitor_state.pkl"): os.remove("monitor_state.pkl")
        run_count = 0
        with open("run_count.txt", "w") as f:
            f.write(str(run_count))
    else:
        save_neuron_state(neurons, "network_state.pkl")
        save_monitor_state(monitor, "monitor_state.pkl")

    print("[Step 11] Appending summary to log file...")
    log_filename = get_log_filename()

    def convert_to_serializable(obj):
        """Convert numpy types to standard Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj

    json_summary = {
        "clusters": convert_to_serializable(summary),
        "input_signal": float(input_signal),
        "goal": goal,
        "total_neurons": int(num_neurons),
        "active_neurons": int(len([n for n in neurons if n.active])),
        "iteration": monitor.iteration,
        "summary": convert_to_serializable(monitor.summarize())
    }

    append_run_log(json_summary, log_filename)

    print("Experiment completed successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train bio-inspired neural network.")
    parser.add_argument('-l', '--loops', type=int, default=globals().get('TRAINING_LOOPS', 200), help='Number of experiment loops to run')
    args = parser.parse_args()
    for i in range(args.loops):
        print(f"\n=== Experiment Loop {i+1}/{args.loops} ===")
        if i == 0:
            load_config(granular=True, training=False)
            train_green_detection()
        else:
            load_config(granular=True, training=True)
            # Interactive CLI for manual override of training parameters
            import importlib
            import config_training
            print("\n--- Manual Review: Granular Training Parameters ---")
            for k in dir(config_training):
                if k.isupper() and not k.startswith('__'):
                    print(f"{k}: {getattr(config_training, k)}")
            print("\nYou may now manually edit config_training.py to adjust any parameter.")
            input("Press Enter when ready to continue with the next training loop...")
            importlib.reload(config_training)
            load_config(granular=True, training=True)  # Reload globals with any changes
            train_green_detection()