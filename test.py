import torch
from torch.utils.data import DataLoader
from law_data import Law_Dataset
from model_paraformer import Model_Paraformer
from processing_data import preprocessor_batch
from eval_metrics import eval_model
from raw_data import df_create_data_training

import my_env


def eval():
    # Load the test dataset
    print("eval Model")
    df_test = df_create_data_training(
        my_env.PATH_TO_PUBLIC_TRAIN, my_env.PATH_TO_CORPUS)
    test_dataset = Law_Dataset(df_test)
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, num_workers=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    model = Model_Paraformer().to(device)
    model.load_state_dict(torch.load('paraformer2.pth'))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    # Evaluate the model on the test dataset
    test_loss, accuracy, precision, recall, f2_score = eval_model(
        test_dataloader, model, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F2 Score: {f2_score:.4f}")


def test():
    # Load the test dataset
    df_test = df_create_data_training(
        my_env.PATH_TO_PUBLIC_TEST, my_env.PATH_TO_CORPUS)
    test_dataset = Law_Dataset(df_test)
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, num_workers=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    model = Model_Paraformer().to(device)
    model.load_state_dict(torch.load('paraformer.pth'))
    model.eval()

    # Test the model on the test dataset
    predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            questions, articles, _ = batch
            questions = questions.to(device)
            articles = articles.to(device)

            output = model.forward(questions, articles)

            _, predicted = torch.max(output, dim=1)
            predictions.extend(predicted.tolist())

    # Do something with the predictions
    print(predictions)


if __name__ == "__main__":
    test()
