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
    print("Starting green detection experiment...")
    try:
        with open("run_count.txt", "r") as f:
            run_count = int(f.read().strip())
    except Exception:
        run_count = 0
    run_count += 1
    with open("run_count.txt", "w") as f:
        f.write(str(run_count))

    goal = "detect_green"
    num_neurons = 10

    print("[Step 1] Loading or initializing neurons and monitor...")
    neurons = load_neuron_state("network_state.pkl")
    if neurons is None:
        print("  Creating new neural network...")
        neurons = [Neuron(position=(i,), specialty=np.random.uniform(10, 15)) for i in range(num_neurons)]
        for i in range(len(neurons)):
            if i > 0:
                neurons[i].add_neighbor(neurons[i-1])
            if i < len(neurons) - 1:
                neurons[i].add_neighbor(neurons[i+1])
    else:
        print(f"  Loaded {len(neurons)} neurons from state.")

    monitor = load_monitor_state("monitor_state.pkl")
    print("  Creating new monitor (always references current neurons)...")
    monitor = NeuronEnsembleMonitor(neurons)

    print("[Step 2] Loading and processing input pixel...")
    import random
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    pixel = load_pixel((r, g, b))
    print(f"  Input pixel: {pixel}")

    goal_options = ["detect_red", "detect_green", "detect_blue"]
    goal = random.choice(goal_options)
    print(f"[Step 2b] Goal for this run: {goal}")

    print("[Step 4] Setting goal for all neurons...")
    for neuron in neurons:
        neuron.receive_goal(goal)

    print("[Step 5] Calculating input signal from selected color component...")
    color_map = {"detect_red": 0, "detect_green": 1, "detect_blue": 2}
    color_idx = color_map[goal]
    input_signal = pixel[color_idx] / 255.0
    print(f"  Input signal strength: {input_signal}")

    print("[Step 6] Evaluating activation for all neurons (with weight noise)...")
    for idx, neuron in enumerate(neurons):
        noise = random.uniform(-0.05, 0.05)
        neuron.weight = min(1.0, max(0.0, neuron.weight + noise))
        neuron.evaluate_activation(input_signal)
        print(f"    Neuron {idx}: specialty={neuron.specialty:.4f}, weight={neuron.weight:.4f} (noise={noise:+.4f}), active={neuron.active}")

    print("[Step 7] Neurons sending messages to neighbors (with message noise)...")
    for idx, neuron in enumerate(neurons):
        original_send_message = neuron.send_message
        def noisy_send_message():
            if neuron.active and neuron.has_fired_this_round:
                for neighbor in neuron.neighbors:
                    if random.random() < 0.1:
                        continue
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

    specialties = [neuron.specialty for neuron in neurons]
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train bio-inspired neural network.")
    parser.add_argument('-l', '--loops', type=int, default=1, help='Number of experiment loops to run')
    args = parser.parse_args()

    last_neurons = None
    last_monitor = None
    for i in range(args.loops):
        print(f"\n=== Experiment Loop {i+1}/{args.loops} ===")
        train_green_detection()
        # After each loop, reload the latest state
        last_neurons = load_neuron_state("network_state.pkl")
        last_monitor = load_monitor_state("monitor_state.pkl")

    # Always save the final state after all loops
    if last_neurons is not None:
        save_neuron_state(last_neurons, "network_state.pkl")
    if last_monitor is not None:
        save_monitor_state(last_monitor, "monitor_state.pkl")
