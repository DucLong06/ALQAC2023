import torch
from tqdm import tqdm
from src.task1.model_paraformer import Model_Paraformer


def calculate_accuracy(correct_predictions, total_questions) -> float:
    return correct_predictions / total_questions


def calculate_precision(true_positive, false_positive) -> float:
    return true_positive / (true_positive + false_positive)


def calculate_recall(true_positive, false_negative) -> float:
    return true_positive / (true_positive + false_negative)


def calculate_f2_score(precision, recall, beta=2) -> float:
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)


def eval_model(data_loader, model: Model_Paraformer):
    print("eval model")
    model.eval()
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    total_loss = 0.0
    with torch.no_grad():
        for query, article, label in tqdm(data_loader):
            label = torch.tensor(label).cpu()
            output = model.predict(query, article)
            # total_loss += model.criterion(output, label).item()
            
            total += label.size(0)
            correct += output.eq(label).sum().item()
            true_positive += (output.eq(1) & label.eq(1)).sum().item()
            # true_negative += (output.eq(0) & label.eq(0)).sum().item()
            false_positive += (output.eq(1) & label.eq(0)).sum().item()
            false_negative += (output.eq(0) & label.eq(1)).sum().item()
            
    loss = total_loss/len(data_loader)
    accuracy = calculate_accuracy(correct, total)
    precision = calculate_precision(true_positive, false_positive)
    recall = calculate_recall(true_positive, false_negative)
    f2_score = calculate_f2_score(precision, recall)

    return loss,accuracy, precision, recall, f2_score
