import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from eval_metrics import eval_model
from law_data import Law_Dataset
from model_paraformer import Model_Paraformer
from raw_data import df_create_data_training
import my_env
from tqdm import tqdm



def preprocessor_batch(batch):
    questions, articles, relevants = zip(*batch)

    max_article_length = max(len(article) for article in articles)

    padded_articles = [article + [''] * (max_article_length - len(article)) if len(
        article) < max_article_length else article for article in articles]

    return questions, padded_articles, relevants


def train():
    df_train = df_create_data_training(
        my_env.PATH_TO_QUESTION_ALL, my_env.PATH_TO_CORPUS_ALL, top_bm25=10)
    df_train = Law_Dataset(df_train)

    # Split dataset into train and test set
    train_size = int(0.8 * len(df_train))
    test_size = len(df_train) - train_size
    train_dataset, test_dataset = random_split(
        df_train, [train_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=1, num_workers=4, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, num_workers=4, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model_Paraformer().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, eps=1e-8)

    max_epochs = 10
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs}")
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            questions, articles, relevants = batch
            logits = model.forward(questions, articles)
            relevants = relevants.to(device)

            loss = model.criterion(logits, relevants)

            loss.backward()
            optimizer.step()
            # scheduler.step(loss)
            total_loss += loss.item()

        train_loss = total_loss / len(train_dataloader)
        print(f"Loss: {train_loss:.4f}")
        print("================================================")

    print('Training finished.')

    # Evaluate the model on the test set
    test_accuracy, test_precision, test_recall, test_f2_score = eval_model(
        test_dataloader, model)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F2 Score: {test_f2_score:.4f}")

    # Save the model
    torch.save(model.state_dict(), 'model_new.pth')
    print("Done !!!")


if __name__ == "__main__":
    train()
