from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def compute_metrics(p):
    # Get the predicted labels by taking the argmax over the logits
    predictions = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    
    # Compute F1 score (for binary classification)
    f1 = f1_score(labels, predictions, average="binary")
    
    # Compute accuracy
    accuracy = accuracy_score(labels, predictions)
    
    return {"f1": f1, "accuracy": accuracy}
