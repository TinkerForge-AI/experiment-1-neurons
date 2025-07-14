import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from neuron import Neuron
from neuron_ensemble_monitor import NeuronEnsembleMonitor
from state_mgr.neuron_state_manager import save_neuron_state, load_neuron_state
from state_mgr.monitor_state_manager import save_monitor_state, load_monitor_state
from mappings.data_loader import load_pixel
import numpy as np

LOGS_DIR = "logs"

def plot_specialty_distribution(specialties):
    """
    Plot and save the specialty distribution histogram ONCE after training completes.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(specialties, bins=20, range=(min(specialties), max(specialties)), color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Neuron Specialty Distribution (Final)")
    plt.xlabel("Specialty Value")
    plt.ylabel("Neuron Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    # Save to training_logs folder
    if not os.path.exists("training_logs"):
        os.makedirs("training_logs")
    plt.savefig("training_logs/specialty_distribution_final.png")
    plt.close()

def get_log_filename():
    today = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    return os.path.join(LOGS_DIR, f"{today}.json")

def append_run_log(summary, log_filename):
    try:
        if os.path.exists(log_filename):
            try:
                with open(log_filename, "r") as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Corrupted log file {log_filename}, creating new one")
                logs = []
        else:
            logs = []
        summary["timestamp"] = datetime.now().isoformat()
        logs.append(summary)
        with open(log_filename, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"Log appended to {log_filename}")
    except Exception as e:
        print(f"Error saving log: {e}")
        backup_filename = log_filename.replace('.json', '_backup.json')
        try:
            with open(backup_filename, "w") as f:
                json.dump([summary], f, indent=2)
            print(f"Saved as backup: {backup_filename}")
        except Exception as backup_error:
            print(f"Failed to save backup: {backup_error}")

def train_green_detection():
    import matplotlib.pyplot as plt
    # ...existing code...
    def plot_specialty_distribution(specialties, run_count):
        plt.figure(figsize=(8, 4))
        plt.hist(specialties, bins=20, range=(min(specialties), max(specialties)), color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f"Neuron Specialty Distribution (Run {run_count})")
        plt.xlabel("Specialty Value")
        plt.ylabel("Neuron Count")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"specialty_distribution_run_{run_count}.png")
        plt.close()
    # --- Output capture setup ---
    import sys
    logs_dir = "training_logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    output_filename = os.path.join(logs_dir, "training_run.txt")
    # Aggregate all loops into one file, append mode
    output_file = open(output_filename, 'a')
    old_stdout = sys.stdout
    sys.stdout = output_file
    print(f"\n[INFO] Capturing output to {output_filename}")
    print("Starting green detection experiment...")
    try:
        with open("run_count.txt", "r") as f:
            run_count = int(f.read().strip())
    except Exception:
        run_count = 0
    run_count += 1
    with open("run_count.txt", "w") as f:
        f.write(str(run_count))

    # Use NUM_NEURONS from config for network size
    try:
        import config
        NUM_NEURONS = getattr(config, 'NUM_NEURONS', 9)
        color_labels = getattr(config, 'COLOR_LABELS', ["red", "green", "blue"])
        color_rgbs = getattr(config, 'COLOR_RGBS', [(255,0,0), (0,255,0), (0,0,255)])
    except Exception:
        from config import NUM_NEURONS, COLOR_LABELS, COLOR_RGBS
        color_labels = COLOR_LABELS
        color_rgbs = COLOR_RGBS
    num_neurons = NUM_NEURONS

    print("[Step 1] Loading or initializing neurons and monitor...")
    neurons = load_neuron_state("network_state.pkl")
    if neurons is None:
        print("  Creating new neural network...")
        # Randomize or evenly distribute initial specialties across the full range
        import random
        neurons = []
        for i in range(num_neurons):
            initial_specialty = random.uniform(1.0, 10.0)
            neurons.append(Neuron(position=(i,), specialty=initial_specialty))
        
        # Create robust neighbor connections - each neuron has 2-4 neighbors
        for i in range(len(neurons)):
            # Linear neighbors
            if i > 0:
                neurons[i].add_neighbor(neurons[i-1])
            if i < len(neurons) - 1:
                neurons[i].add_neighbor(neurons[i+1])
            
            # Add some skip connections for better communication
            if i >= 2:
                neurons[i].add_neighbor(neurons[i-2])
            if i < len(neurons) - 2:
                neurons[i].add_neighbor(neurons[i+2])
        
        print(f"  Created {len(neurons)} neurons with distributed specialties")
        for i, n in enumerate(neurons):
            print(f"    Neuron {i}: specialty={n.specialty:.2f}, neighbors={len(n.neighbors)}")
    else:
        print(f"  Loaded {len(neurons)} neurons from state.")

    # Assign neurons to clusters
    num_clusters = 3  # Or from config
    cluster_size = max(1, len(neurons) // num_clusters)
    clusters = []
    for i in range(num_clusters):
        cluster_neurons = neurons[i*cluster_size:(i+1)*cluster_size]
        from cluster import Cluster
        clusters.append(Cluster(cluster_neurons, cluster_id=f"C{i}"))
    
    # Create controller and monitor with proper integration
    from cluster_controller import ClusterController
    controller = ClusterController(clusters, controller_id="TrainController")
    monitor = NeuronEnsembleMonitor(neurons, clusters=clusters, controllers=[controller])
    
    print(f"[Step 1b] Created controller with cluster assignments: {controller.cluster_assignments}")
    print(f"[Step 1c] Monitor tracking {len(monitor.neurons)} neurons, {len(monitor.clusters)} clusters, {len(monitor.controllers)} controllers")

    print("[Step 2] Loading and processing input pixel...")
    import random
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    pixel = load_pixel((r, g, b))
    print(f"  Input pixel: {pixel}")

    goal_idx = random.randint(0, len(color_labels) - 1)
    goal = f"detect_{color_labels[goal_idx]}"
    print(f"[Step 2b] Goal for this run: {goal}")

    print("[Step 4] Setting goal for all neurons...")
    for neuron in neurons:
        neuron.receive_goal(goal)

    print("[Step 5] Calculating input signal from selected color component...")
    # For 12 colors, use the max channel as signal, or a custom mapping
    input_signal = max(pixel) / 255.0
    print(f"  Input signal strength: {input_signal}")

    print("[Step 6] Evaluating activation for all clusters and applying error-driven feedback...")
    is_initial_round = run_count <= 3
    print(f"  Run count: {run_count}, Initial round: {is_initial_round}")
    # Set training round on clusters and neurons
    for cluster in clusters:
        cluster.training_round = run_count
        for neuron in cluster.neurons:
            neuron.cluster = cluster
            neuron.training_round = run_count
            noise = random.uniform(-0.05, 0.05)
            # For new system, neuron.weight is a dict; update all weights
            if hasattr(neuron, 'weight') and isinstance(neuron.weight, dict):
                for k in neuron.weight:
                    neuron.weight[k] = min(2.0, max(0.1, neuron.weight[k] + noise))
    # Use controller to route signal and get feedback
    true_label = goal.replace("detect_", "")
    feedback_result = controller.route_signal(input_signal, goal=goal, true_label=true_label)
    # Apply error-driven learning: weaken winner weights if wrong, strengthen correct neurons
    for cluster in clusters:
        try:
            from config import SPECIALTY_LEARNING_RATE, WEIGHT_LEARNING_RATE, SPECIALTY_CLAMP_MIN, SPECIALTY_CLAMP_MAX
        except ImportError:
            SPECIALTY_LEARNING_RATE, WEIGHT_LEARNING_RATE = 0.1, 0.05
            SPECIALTY_CLAMP_MIN, SPECIALTY_CLAMP_MAX = 1.0, 10.0
        for neuron in cluster.neurons:
            neuron_guess = neuron.get_preferred_color()
            neuron.last_guess = neuron_guess
            correct = (neuron_guess == true_label)
            # If neuron is winner and wrong, apply reduced penalty
            if neuron.is_winner and not correct:
                for k in neuron.weight:
                    neuron.weight[k] *= (1.0 - WEIGHT_LEARNING_RATE/2)
                neuron.specialty -= SPECIALTY_LEARNING_RATE/2
            # If neuron is winner and correct, apply much stronger reward
            if neuron.is_winner and correct:
                for k in neuron.weight:
                    neuron.weight[k] *= (1.0 + 2*WEIGHT_LEARNING_RATE)
                neuron.specialty += 2*SPECIALTY_LEARNING_RATE
            # If neuron is not winner but would have been correct, apply stronger reward
            if not neuron.is_winner and correct:
                for k in neuron.weight:
                    neuron.weight[k] *= (1.0 + WEIGHT_LEARNING_RATE)
                neuron.specialty += SPECIALTY_LEARNING_RATE
            # Clamp specialty
            neuron.specialty = max(SPECIALTY_CLAMP_MIN, min(SPECIALTY_CLAMP_MAX, neuron.specialty))
            status = "INITIAL_ACTIVE" if is_initial_round else ("ACTIVE" if neuron.active else "INACTIVE")
            print(f"    Neuron {neuron.position[0]}: specialty={neuron.specialty:.4f}, status={status}, guess={neuron_guess}, correct?={'yes' if correct else 'no'}, winner={neuron.is_winner}")
    # Apply controller guidance for complementary specialization
    controller.ensure_complementary_specialization()

    print("[Step 7] Neurons sending messages to neighbors...")
    total_messages_sent = 0
    for idx, neuron in enumerate(neurons):
        messages_sent = neuron.send_message(training_round=run_count)
        total_messages_sent += messages_sent
        
        # Add some additional random messaging for exploration
        if random.random() < 0.1 and neuron.neighbors:
            neighbor = random.choice(neuron.neighbors)
            specialty_diff = abs(neuron.specialty - neighbor.specialty)
            msg_type = 'excite' if specialty_diff < 2.0 else 'inhibit'
            neighbor.receive_message({
                'type': msg_type, 
                'from': neuron.position,
                'training_round': run_count
            })
            total_messages_sent += 1
            
    print(f"  Total messages sent: {total_messages_sent}")

    print("[Step 8] Neuron learning step (with learning noise)... [REWRITE START]")
    neuron_changes = []

    for idx, neuron in enumerate(neurons):
        # Track before/after for diagnostics
        old_specialty = neuron.specialty
        old_weight = dict(neuron.weight) if isinstance(neuron.weight, dict) else neuron.weight
        old_active = neuron.active
        learning_noise = random.uniform(-0.1, 0.1)
        neuron_guess = getattr(neuron, 'last_guess', None)
        true_label = goal.replace("detect_", "")

        # Learning logic
        # If neuron is active, apply noise and reinforce specialty
        if neuron.active:
            neuron.specialty += learning_noise
            # Optionally reinforce correct guess
            if neuron_guess == true_label:
                neuron.specialty += 0.05
            else:
                neuron.specialty -= 0.05
            neuron.specialty = max(1.0, min(10.0, neuron.specialty))
            neuron.evaluate_activation()
        else:
            # Inactive neurons drift toward input signal, plus noise
            try:
                from config import SPECIALTY_LEARNING_RATE, SPECIALTY_INIT_MIN, SPECIALTY_INIT_MAX
            except ImportError:
                SPECIALTY_LEARNING_RATE, SPECIALTY_INIT_MIN, SPECIALTY_INIT_MAX = 0.05, 1.0, 10.0
            neuron.specialty += SPECIALTY_LEARNING_RATE * (input_signal - neuron.specialty) + learning_noise
            neuron.specialty = max(SPECIALTY_INIT_MIN, min(SPECIALTY_INIT_MAX, neuron.specialty))

        # Weight update: reinforce correct guesses, punish incorrect
        if isinstance(neuron.weight, dict):
            try:
                from config import WEIGHT_LEARNING_RATE, WEIGHT_RANGE
            except ImportError:
                WEIGHT_LEARNING_RATE, WEIGHT_RANGE = 0.02, (0.1, 2.0)
            for k in neuron.weight:
                if neuron_guess == true_label:
                    neuron.weight[k] *= (1.0 + WEIGHT_LEARNING_RATE)
                else:
                    neuron.weight[k] *= (1.0 - WEIGHT_LEARNING_RATE)
                neuron.weight[k] = min(WEIGHT_RANGE[1], max(WEIGHT_RANGE[0], neuron.weight[k]))
            weight_before = np.mean(list(old_weight.values()))
            weight_after = np.mean(list(neuron.weight.values()))
        else:
            if neuron_guess == true_label:
                neuron.weight *= 1.02
            else:
                neuron.weight *= 0.98
            neuron.weight = min(2.0, max(0.1, neuron.weight))
            weight_before = old_weight
            weight_after = neuron.weight

        specialty_change = neuron.specialty - old_specialty
        weight_change = weight_after - weight_before
        active_change = int(neuron.active) - int(old_active)

        neuron_changes.append({
            'idx': idx,
            'specialty_before': old_specialty,
            'specialty_after': neuron.specialty,
            'specialty_change': specialty_change,
            'weight_before': weight_before,
            'weight_after': weight_after,
            'weight_change': weight_change,
            'active_before': old_active,
            'active_after': neuron.active,
            'active_change': active_change,
            'iteration': neuron.iteration,
            'learning_noise': learning_noise
        })

    print("\nNeuron Change Summary Table: [REWRITE]")
    for change in neuron_changes:
        print(f"Idx: {change['idx']:>3}\n"
            f"  Specialty (before->after) [Δ]: {change['specialty_before']:.4f}->{change['specialty_after']:.4f} [{change['specialty_change']:+.4f}]\n"
            f"  Weight (before->after) [Δ]: {change['weight_before']:.4f}->{change['weight_after']:.4f} [{change['weight_change']:+.4f}]\n"
            f"  Active (before->after) [Δ]: {str(change['active_before']):>5}->{str(change['active_after']):>5} [{change['active_change']:+d}]\n"
            f"  Iteration: {change['iteration']:>3}\n"
            f"  Learning noise: {change['learning_noise']:+.4f}\n"
            "----------------------------------------")
    print("[Step 8] Neuron learning step (with learning noise)... [REWRITE END]")

    print("[Step 9] Logging activity and generating summary...")
    monitor.log_activity(input_label=goal.replace("detect_", ""))
    summary = monitor.summarize()
    monitor.print_human_summary()
    print("[Step 9c] Controller state:", controller.get_state())

    specialties = [neuron.specialty for neuron in neurons]
    # Log specialty mean and variance for convergence monitoring
    specialty_mean = np.mean(specialties)
    specialty_var = np.var(specialties)
    print(f"[Step 9a] Specialty mean: {specialty_mean:.4f}, variance: {specialty_var:.4f}")
    csv_header = "run,specialty_0,specialty_1,specialty_2,specialty_3,specialty_4,specialty_5,specialty_6,specialty_7,specialty_8,specialty_9\n"
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
    # Membrane potential reset for all neurons after each round
    try:
        from config import MEMBRANE_RESET
    except ImportError:
        MEMBRANE_RESET = True
    if MEMBRANE_RESET:
        for neuron in neurons:
            neuron.membrane_potential = 0.0

    print("[Step 11] Saving state for next run...")
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
    # --- Restore stdout ---
    sys.stdout = old_stdout
    output_file.close()
    print(f"[INFO] Training run output saved to {output_filename}")

if __name__ == "__main__":
    import argparse
    from config import TRAINING_LOOPS
    parser = argparse.ArgumentParser(description="Train bio-inspired neural network.")
    parser.add_argument('-l', '--loops', type=int, default=TRAINING_LOOPS, help='Number of experiment loops to run')
    parser.add_argument('-rm', '--remove-model', action='store_true', help='Remove model .pkl files and start from scratch')
    args = parser.parse_args()

    if args.remove_model:
        print("[INFO] Removing model state files: network_state.pkl, monitor_state.pkl")
        if os.path.exists("network_state.pkl"):
            os.remove("network_state.pkl")
            print("  Removed network_state.pkl")
        if os.path.exists("monitor_state.pkl"):
            os.remove("monitor_state.pkl")
            print("  Removed monitor_state.pkl")
        print("[INFO] Model state reset. Training will start from scratch.")

    last_neurons = None
    last_monitor = None
    import importlib
    import config
    try:
        for i in range(args.loops):
            print(f"\n=== Experiment Loop {i+1}/{args.loops} ===")
            # Interactive CLI for manual override of training parameters
            if i > 0:
                print("\n--- Manual Review: Granular Training Parameters ---")
                for k in dir(config):
                    if k.isupper() and not k.startswith('__'):
                        print(f"{k}: {getattr(config, k)}")
                print("\nYou may now manually edit config.py to adjust any parameter.")
                input("Press Enter when ready to continue with the next training loop...")
                importlib.reload(config)
            train_green_detection()
            # After each loop, reload the latest state
            last_neurons = load_neuron_state("network_state.pkl")
            last_monitor = load_monitor_state("monitor_state.pkl")
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user (Ctrl+C). Output file closed.")

    # Always save the final state after all loops
    if last_neurons is not None:
        # After all training, plot specialty distribution ONCE
        specialties = [neuron.specialty for neuron in last_neurons]
        plot_specialty_distribution(specialties)
        save_neuron_state(last_neurons, "network_state.pkl")
    if last_monitor is not None:
        save_monitor_state(last_monitor, "monitor_state.pkl")
