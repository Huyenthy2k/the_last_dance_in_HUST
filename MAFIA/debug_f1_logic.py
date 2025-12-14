import torch as th
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def compute_direction_metrics_custom(
    direction_logits: th.Tensor,  # (B, T_m, 3)
    direction_labels: th.Tensor,  # (B, T_m)
):
    # Get predictions
    preds = th.argmax(direction_logits, dim=-1)  # (B, T_m)

    # Flatten
    preds_flat = preds.view(-1).cpu().numpy()
    labels_flat = direction_labels.view(-1).cpu().numpy()

    # Accuracy
    accuracy = (preds_flat == labels_flat).mean()

    # Per-class F1 scores
    f1_scores = {}
    class_names = ["bear", "side", "bull"]

    for class_id, class_name in enumerate(class_names):
        # Binary: This class vs rest
        pred_binary = (preds_flat == class_id).astype(int)
        label_binary = (labels_flat == class_id).astype(int)

        # TP, FP, FN
        tp = ((pred_binary == 1) & (label_binary == 1)).sum()
        fp = ((pred_binary == 1) & (label_binary == 0)).sum()
        fn = ((pred_binary == 0) & (label_binary == 1)).sum()

        # Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        f1_scores[f"f1_{class_name}"] = float(f1)

    # Macro F1
    f1_macro = np.mean([f1_scores[f"f1_{name}"] for name in class_names])

    return {
        "accuracy": float(accuracy),
        **f1_scores,
        "f1_macro": float(f1_macro),
    }

def verify_logic():
    print("Verifying F1 Logic...")
    
    # 1. Create Dummy Data
    B, T = 10, 5
    # Random logits
    logits = th.randn(B, T, 3) 
    # Random targets (0, 1, 2)
    labels = th.randint(0, 3, (B, T))
    
    # Custom
    custom_metrics = compute_direction_metrics_custom(logits, labels)
    
    # Sklearn Reference
    preds_flat = th.argmax(logits, dim=-1).view(-1).numpy()
    labels_flat = labels.view(-1).numpy()
    
    sk_acc = accuracy_score(labels_flat, preds_flat)
    sk_f1_macro = f1_score(labels_flat, preds_flat, average='macro')
    sk_f1_none = f1_score(labels_flat, preds_flat, average=None, labels=[0, 1, 2])
    # Note: average=None returns array [f1_0, f1_1, f1_2]
    # If a class is missing in pred/target, sklearn might warn or return 0 depending on params.
    # To match our logic (default 0), we assume all classes exist or we handle 0.
    
    print("\n--- Comparison ---")
    print(f"Accuracy: Custom={custom_metrics['accuracy']:.6f} vs Sklearn={sk_acc:.6f}")
    print(f"F1 Macro: Custom={custom_metrics['f1_macro']:.6f} vs Sklearn={sk_f1_macro:.6f}")
    
    classes = ["bear", "side", "bull"]
    for i, name in enumerate(classes):
        # We need to handle case where sklearn might skip a class if not present?
        # sk_f1_none is aligned with `labels` arg.
        sk_val = sk_f1_none[i]
        custom_val = custom_metrics[f"f1_{name}"]
        print(f"F1 {name.title()}: Custom={custom_val:.6f} vs Sklearn={sk_val:.6f}")
        
    # Check for discrepancies
    assert np.isclose(custom_metrics['accuracy'], sk_acc), "Accuracy Mismatch"
    assert np.isclose(custom_metrics['f1_macro'], sk_f1_macro), "Macro F1 Mismatch"
    
    print("\n✅ Logic verified correctly against Sklearn.")

if __name__ == "__main__":
    verify_logic()
