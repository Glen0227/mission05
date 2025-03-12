import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def psnr(mse):
    i_max = 255.0
    psnr_value = 10 * torch.log10(i_max**2/mse)
    return psnr_value


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.early_stop = False
        self.val_loss_min = float('inf')
        self.best_model_wts = None

        self.counter = 0

    def __call__(self, val_loss, model):
        if val_loss < self.val_loss_min - self.delta:
            self.val_loss_min = val_loss
            self.best_model_wts = model.state_dict()
            self.counter = 0

        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
    

def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses = []
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            losses.append(loss.item())

            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = sum(losses) / len(losses)
        accuracy = correct / total

        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')

    return avg_loss, accuracy, precision, recall, f1