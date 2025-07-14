# === Neuron Dynamics ===
LEAK_RATE = 0.08                 # Membrane potential decay per round
REFRACTORY_PERIOD = 1            # Rounds neuron must wait before firing again
# === Non-winner neuron firing ===
NONWINNER_FIRE_PROB = 0.15         # Probability that a non-winner neuron fires
NONWINNER_INFLUENCE_SCALE = 0.3    # Relative influence of non-winner neuron output (vs. winner)

# Unified experiment configuration for neural network

# === Core Experiment Parameters ===
NUM_NEURONS = 9               # Total number of neurons in the network
NUM_CLUSTERS = 3                # Number of clusters/groups
TRAINING_LOOPS = 5              # Number of full training cycles
TEST_SAMPLES = 10               # Number of test samples per evaluation
MEMBRANE_RESET = True           # Whether to reset membrane potential after each round

# === Cluster Management ===
CLUSTER_REBALANCE_INTERVAL = 20 # Number of training rounds between cluster rebalancing
CLUSTER_SPECIALTIES = [2, 6, 10]# Target specialty values for each cluster

# === Color/Label Parameters ===
COLOR_LABELS = ["red", "green", "blue"] # Output labels
COLOR_RGBS = [(255,0,0), (0,255,0), (0,0,255)] # RGB values for each label

# === Neuron Initialization ===
SPECIALTY_INIT_MIN = 1.0        # Minimum initial specialty value
SPECIALTY_INIT_MAX = 30.0       # Maximum initial specialty value
SPECIALTY_INIT_RANGE = SPECIALTY_INIT_MAX - SPECIALTY_INIT_MIN # Range for specialty initialization
WEIGHT_RANGE = (0.3, 1.0)       # Initial synaptic weight range
CONFIDENCE_INIT = 0.5           # Initial confidence value for each neuron

# === Learning Rates ===
SPECIALTY_LEARNING_RATE = SPECIALTY_INIT_RANGE * 0.3   # Learning rate for specialty updates
WEIGHT_LEARNING_RATE = 0.15     # Learning rate for synaptic weights
CONFIDENCE_UPDATE_CORRECT = 0.25 # Confidence increase for correct prediction
CONFIDENCE_UPDATE_INCORRECT = 0.1 # Confidence decrease for incorrect prediction

# === Training Progression ===
EARLY_TRAINING_ROUNDS = 50      # Rounds with gentler confidence penalties
CONFIDENCE_SPECIALIZATION_THRESHOLD = 0.85 # Threshold for sending teach messages

# === Message/Communication ===
MESSAGE_NOISE_PROB = 0.08        # Probability of noise in messages
MESSAGE_FLIP_PROB = 0.12         # Probability of message bit flip
MESSAGE_CONFIDENCE_THRESHOLD = 0.72 # Confidence required to send message

# === Confidence/Reach Parameters ===
CONFIDENCE_REACH_THRESHOLD = 0.2499 # If neuron confidence < this, it reaches for help
CONFIDENCE_REACH_RATE = 0.3       # How much a low-confidence neuron adopts from a confident peer/controller
CONFIDENCE_TOTAL_MIN = NUM_NEURONS # Minimum total confidence for the system
CONFIDENCE_MAX = 1.0               # Maximum confidence per neuron

# === Specialty Learning ===
SPECIALTY_REPULSION_STRENGTH = 0.2 # Strength of specialty repulsion between neurons
SPECIALTY_REPULSION_DISTANCE = 2.0 # Minimum distance for repulsion to apply
SPECIALTY_CLAMP_MIN = SPECIALTY_INIT_MIN # Minimum allowed specialty value
SPECIALTY_CLAMP_MAX = SPECIALTY_INIT_MAX # Maximum allowed specialty value
WEIGHT_NOISE_RANGE = (-0.05, 0.05)  # Range for random noise added to weights
LEARNING_NOISE_RANGE = (-0.3, 0.3)  # Range for random noise added during learning
RANDOM_FLIP_PROB = 0.15             # Probability of random specialty/weight flip

# === Override/Activation ===
OVERRIDE_CONFIDENCE_DELTA = 0.1    # Confidence penalty/bonus for forced activation override
OVERRIDE_CONFIDENCE_THRESHOLD = 0.75 # Neuron only overrides cluster if confidence >= threshold

# === Adaptive Learning Rate ===
CONCURRENT_ANSWER_THRESHOLD = 5     # Consecutive correct guesses before reducing learning rate
SPECIALTY_LEARNING_RATE_MIN = 0.05  # Minimum allowed specialty learning rate after stabilization
CONFIDENCE_UPDATE_CORRECT_MIN = 0.05 # Minimum confidence update for correct answers

# === Activation ===
ACTIVATION_THRESHOLD = 0.3          # Threshold for neuron activation

# === Miscellaneous ===
# Add more as needed for manual override or future features
