from ConfigSpace import Configuration, ConfigurationSpace
import numpy as np
from methods import CDET
from smac import HyperparameterOptimizationFacade, Scenario
from strlearn.streams import StreamGenerator

def test_detections(config: Configuration, seed=None) -> float:
    # det = CDET(alpha=config["alpha"], ensemble_size=config["ensemble_size"], 
    #            n_replications=config["replications"], stat_proba=config["stat_proba"], 
    #            neck_width=10, th=config["th"])
    
    det = CDET(alpha=config["alpha"], ensemble_size=12, 
               n_replications=12, stat_proba=75, 
               neck_width=10, th=config["th"])
    
    n_chunks= 250
    n_drifts = 10
    stream = StreamGenerator(n_chunks=n_chunks,
                            chunk_size=100,
                            n_drifts=n_drifts,
                            n_classes=3,
                            n_features=40,
                            n_informative=7,
                            n_redundant=0,
                            n_repeated=0,
                            concept_sigmoid_spacing=900,
                            random_state=seed)

    
    detections = []
    for i in range(n_chunks):
        X, y = stream.get_chunk()
        det.process(X)
        
        if det._is_drift:
            detections.append(i)
    
    return np.abs(n_drifts - len(detections))


configspace = ConfigurationSpace({
    "alpha": (0.001, 0.3),
    # "ensemble_size": (1,100),
    # "replications": (1, 100),
    # "stat_proba": (10,200),
    "th": (0.01, 0.9),
    })

# Scenario object specifying the optimization environment
scenario = Scenario(configspace, 
                    deterministic=False, 
                    n_trials=200)

# Use SMAC to find the best configuration/hyperparameters
smac = HyperparameterOptimizationFacade(scenario, test_detections)
incumbent = smac.optimize()
