##################################################
# Evaluation script for CIFAR-100 models
##################################################
# Please download models from the link provided in README.md


from models import get_model
from data_loader import get_data_loaders
import torch
import torch.nn as nn


def evaluate_model(model_type):
    # model_type: "cnn" or "mlp"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    model = get_model(model_type="cnn").to(device)

    # load checkpoint
    model_path = "best_model_cnn.pth"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    # data loader
    _, _, test_loader = get_data_loaders(batch_size=64, model_type=model_type)

    # evaluation
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = loss_sum / total
    accuracy = 100.0 * correct / total

    print(f"{model_type.upper()} Test Loss: {avg_loss:.4f}")
    print(f"{model_type.upper()} Test Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy


# -------------------------------
# Usage
# -------------------------------
if __name__ == "__main__":
    print("Evaluating CNN Model.......")
    evaluate_model("cnn")
    print("\nEvaluating MLP Model.......")
    evaluate_model("mlp")
