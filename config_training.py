# Training configuration for neural network (overrides config.py)

NUM_NEURONS = 9
NUM_CLUSTERS = 3
TRAINING_LOOPS = 200
TEST_SAMPLES = 10

COLOR_LABELS = [
    "red", "green", "blue",
]
COLOR_RGBS = [
    (255,0,0), (0,255,0), (0,0,255),
]
CLUSTER_SPECIALTIES = [2, 6, 10]

# Granular training parameters
ACTIVATION_THRESHOLD = 0.7
WEIGHT_RANGE = (0.3, 1.0)
CONFIDENCE_INIT = 0.5
CONFIDENCE_UPDATE_CORRECT = 0.2
CONFIDENCE_UPDATE_INCORRECT = 0.2
MESSAGE_NOISE_PROB = 0.1
MESSAGE_FLIP_PROB = 0.1
MESSAGE_CONFIDENCE_THRESHOLD = 0.7
SPECIALTY_LEARNING_RATE = 0.5
SPECIALTY_REPULSION_STRENGTH = 0.2
SPECIALTY_REPULSION_DISTANCE = 2.0
SPECIALTY_CLAMP_MIN = 1.0
SPECIALTY_CLAMP_MAX = 10.0
WEIGHT_NOISE_RANGE = (-0.05, 0.05)
LEARNING_NOISE_RANGE = (-0.1, 0.1)
RANDOM_FLIP_PROB = 0.05
OVERRIDE_CONFIDENCE_DELTA = 0.15  # Confidence penalty/bonus for forced activation override
OVERRIDE_CONFIDENCE_THRESHOLD = 0.7  # Neuron only overrides cluster if confidence >= threshold
CONFIDENCE_SPECIALIZATION_THRESHOLD = 0.85
# Confidence equalization and reach parameters
CONFIDENCE_REACH_THRESHOLD = 0.25  # If neuron confidence < this, it reaches for help
CONFIDENCE_REACH_RATE = 0.5        # How much a low-confidence neuron adopts from a confident peer/controller
CONFIDENCE_TOTAL_MIN = NUM_NEURONS # Minimum total confidence for the system
CONFIDENCE_MAX = 1.0               # Maximum confidence per neuron

# Add more as needed for manual override
