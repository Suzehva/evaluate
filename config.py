# config.py

DATASET = None
MODEL = None
SAMPLE_SIZE = None
PROMPTING_T = None
USE_ENSEMBLE = None
ENSEMBLE_SIZE = None
FEWSHOT_SIZE = None
NUM_EXPERTS = None

def initialize(args):
    global DATASET, MODEL, SAMPLE_SIZE, PROMPTING_T, USE_ENSEMBLE, ENSEMBLE_SIZE, FEWSHOT_SIZE, NUM_EXPERTS
    
    DATASET = args.dataset
    MODEL = args.model
    SAMPLE_SIZE = args.sample_size
    PROMPTING_T = args.prompting_type 
    if PROMPTING_T == 'FS' or PROMPTING_T == 'COT_FS':
        FEWSHOT_SIZE = args.prompting_arg
    elif PROMPTING_T == 'TOT':
        NUM_EXPERTS = args.prompting_arg
    USE_ENSEMBLE = args.use_ensemble
    if USE_ENSEMBLE:
        ENSEMBLE_SIZE = args.ensemble_size
