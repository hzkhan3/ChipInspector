import torch

from src.models.classifier import focal_loss

def test_focal_loss():
    logits = torch.tensor([[4.0, -4.0], [0.2, 0.0]])  # one easy, confident example, one hard
    labels = torch.tensor([0, 0])
    ce = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.isclose(focal_loss(logits, labels, gamma=0), ce)  # gamma 0 is plain cross-entropy

    per_example = [focal_loss(logits[i:i + 1], labels[i:i + 1]) / torch.nn.functional.cross_entropy(logits[i:i + 1], labels[i:i + 1])
                   for i in range(2)]
    assert per_example[0] < 0.001 < per_example[1]  # the easy example is down-weighted far more
