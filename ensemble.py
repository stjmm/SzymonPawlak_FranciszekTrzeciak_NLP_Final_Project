import numpy as np
import json
from pathlib import Path
from scipy.optimize import minimize
from sklearn.metrics import log_loss, accuracy_score
from datetime import datetime

from config import Config, MODELS

def optimize_weights(probs_dict, labels):
    model_names = list(probs_dict.keys())
    n_models = len(model_names)
    
    def loss_func(weights):
        normalized_weights = weights / np.sum(weights)
        
        final_probs = np.zeros_like(probs_dict[model_names[0]])
        for i in range(n_models):
            final_probs += normalized_weights[i] * probs_dict[model_names[i]]
            
        return log_loss(labels, final_probs)

    initial_weights = [1.0 / n_models] * n_models
    bounds = [(0, 1)] * n_models
    
    result = minimize(
        loss_func, 
        initial_weights, 
        method='Nelder-Mead', 
        bounds=bounds
    )
    
    final_weights = result.x / np.sum(result.x)
    return {model_names[i]: float(final_weights[i]) for i in range(n_models)}

def ensemble_predict(probs_dict, weights):
    model_names = list(probs_dict.keys())
    final_probs = np.zeros_like(probs_dict[model_names[0]])
    
    for name in model_names:
        final_probs += weights[name] * probs_dict[name]
        
    preds = np.argmax(final_probs, axis=1)
    return preds, final_probs

def run_full_ensemble():
    config = Config()
    output_dir = Path(config.output_dir)
    ensemble_dir = output_dir / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    from data import load_data
    _, labels = load_data(config.data_dir)
    labels = np.array(labels)
    
    probs_dict = {}
    for model_key in MODELS.keys():
        path = output_dir / model_key / "oof_probs.npy"
        if path.exists():
            print(f"Loading OOF...")
            probs_dict[model_key] = np.load(path)
        else:
            print(f"No data.")

    if not probs_dict:
        return

    weights = optimize_weights(probs_dict, labels)
    
    print("Calculated weights")
    for name, w in weights.items():
        print(f"  - {name}: {w:.4f}")

    # 3. Finalne połączenie
    preds, final_probs = ensemble_predict(probs_dict, weights)
    acc = accuracy_score(labels, preds)
    print(f"\n New ensemble accuracy")

    np.save(ensemble_dir / "oof_probs.npy", final_probs)
    
    summary = {
        "model": "Weighted Ensemble",
        "cv_accuracy": acc,
        "fold_accuracies": [acc], # Traktujemy ensemble jako jeden spójny wynik
        "weights": weights,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(ensemble_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    run_full_ensemble()
