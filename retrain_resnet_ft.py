# retrain_resnet_ft.py
from run_experiments import run_experiment
from config import SEED

if __name__ == "__main__":
    print("Re-training resnet_ft with Dropout in head...")
    r = run_experiment(
        "resnet_ft_v2",
        arch="resnet",
        resnet_mode="fine_tune",
        epochs=20,
    )
    print(f"Best val acc: {r['history']['best_val_acc']:.4f}")
    print(f"Macro F1: {r['macro_f1']:.4f}")