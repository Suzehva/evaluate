def initialize(args):
    """
    DATASET: the dataset to use to prompt an LLM
        math, natural_questions 
        (sourced from respectively hendrycks/competition_math and natural_questions on HuggingFace)
    MODEL: the LLM to use
        gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4o
    PROMPTING_T: the prompting technique(s) to use, if any
        "", FS (fewshot), COT (chain of thought), TOT (tree of thought), COT_FS (chain of thought + few shot together)
    SAMPLE_SIZE: the amount queries to an LLM / the amount of datapoints to use
        int
    USE_ENSEMBLE: decides whether ensembling is used 
        True, False
    FEWSHOT_SIZE: amount of fewshot examples used
        int
    NUM_EXPERTS: number of experts to use when using tree of thought
        int
    ENSEMBLE_SIZE: the number of responses generated from the same prompt by an LLM for ensembling
        int
    """
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
