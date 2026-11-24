import torch
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path

class ChipClassifier(torch.nn.Module):
    '''
    ResNet18 backbone with the final layer swapped for a 2-class head
    (0 = nondefective, 1 = defective)
    '''
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)

        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, 2)

    def forward(self, x):
        return self.model(x)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    model = model.to(device)
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # 2 output logits -> predicted class is the index of the larger one
        preds = logits.argmax(dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    model = model.to(device)
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_epoch_loss = running_loss / total
    val_epoch_acc = correct / total

    return val_epoch_loss, val_epoch_acc

def save_model(model, filepath, epoch=None, val_loss=None, val_acc=None):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }, filepath)

def load_model(filepath, device="cpu"):
    # pretrained=False: the saved file already holds all the weights, so no download (works offline)
    model = ChipClassifier(pretrained=False)
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model
