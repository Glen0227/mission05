import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from data.data_loader import data_ready, data_loader_set
from models.GoogLeNet_fine_tune import prep_finetune_GoogLeNet
from utils.metrics import evaluate, EarlyStopping

def train(epochs):
    train_dir, val_dir, test_dir = data_ready()
    train_loader, val_loader, test_loader, class_names = data_loader_set(train_dir, val_dir, test_dir)
 

    model = prep_finetune_GoogLeNet(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=3
)

    early_stopping = EarlyStopping(patience=6, verbose=True)
    print("start training")
    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(images)

            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0) # loss per batch

            predicted = output.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(train_loader.dataset) # draw average by dividing total length
        train_acc = correct/ total


        eval_loss, eval_acc, eval_precision, eval_recall, eval_f1 = evaluate(model, val_loader, criterion, device)
        print(f"scheduler step: {eval_loss}")
        scheduler.step(eval_loss)

        model.train()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Train Acc: {train_acc * 100:.2f}, Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc*100:.2f}")
        print(f"Eval Precision: {eval_precision*100:.2f}, Eval Recall: {eval_recall*100:.2f}, Eval_f1: {eval_f1*100:.2f}")

        if early_stopping(eval_loss, model):
            print("Early stopping triggered.")
            model.load_state_dict(early_stopping.best_model_wts)
            break

    model.eval()

    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy*100:.2f}")
    print(f"Eval Precision: {test_precision*100:.2f}, Eval Recall: {test_recall*100:.2f}, Eval_f1: {test_f1*100:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a GoogLeNet on X-ray datasets from kaggle')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    args = parser.parse_args()
    train(args.epochs) 