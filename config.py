# Experiment configuration for neural network

NUM_NEURONS = 55
NUM_CLUSTERS = 3
TRAINING_LOOPS = 200
TEST_SAMPLES = 10

COLOR_LABELS = [
    "red", "green", "blue", 
    # "yellow", 
    # "cyan", "magenta", "orange", "purple", "pink", "brown", "gray", "black"
]
COLOR_RGBS = [
    (255,0,0), (0,255,0), (0,0,255), 
    # (255,255,0), 
    # (0,255,255), (255,0,255), (255,165,0), (128,0,128), (255,192,203), (165,42,42), (128,128,128), (0,0,0)
]



# Cluster specialties for 3 clusters (red, green, blue)
CLUSTER_SPECIALTIES = [2, 6, 10]  # Example: red=2, green=6, blue=10

# Granular training parameters
ACTIVATION_THRESHOLD = 0.7
WEIGHT_RANGE = (0.3, 1.0)
CONFIDENCE_INIT = 0.5
CONFIDENCE_UPDATE_CORRECT = 0.2
CONFIDENCE_UPDATE_INCORRECT = 0.2
MESSAGE_NOISE_PROB = 0.1
MESSAGE_FLIP_PROB = 0.1
MESSAGE_CONFIDENCE_THRESHOLD = 0.7
SPECIALTY_LEARNING_RATE = 0.3
SPECIALTY_REPULSION_STRENGTH = 0.2
SPECIALTY_REPULSION_DISTANCE = 2.0
SPECIALTY_CLAMP_MIN = 1.0
SPECIALTY_CLAMP_MAX = 10.0
WEIGHT_NOISE_RANGE = (-0.05, 0.05)
LEARNING_NOISE_RANGE = (-0.05, 0.05)
RANDOM_FLIP_PROB = 0.05
OVERRIDE_CONFIDENCE_THRESHOLD = 0.7  # Neuron only overrides cluster if confidence >= threshold
CONFIDENCE_SPECIALIZATION_THRESHOLD = 0.85
