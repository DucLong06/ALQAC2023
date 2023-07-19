import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from bot_telegram import send_telegram_message
from eval_metrics import eval_model
from law_data import Law_Dataset
from model_paraformer import Model_Paraformer
from raw_data import df_create_data_training
import my_env
import asyncio
from tqdm import tqdm


def preprocessor_batch(batch):
    questions, articles, relevants = zip(*batch)

    max_article_length = max(len(article) for article in articles)

    padded_articles = [article + [''] * (max_article_length - len(article)) if len(
        article) < max_article_length else article for article in articles]

    return questions, padded_articles, relevants


def train():
    df_train = df_create_data_training(
        my_env.PATH_TO_QUESTION_ALL, my_env.PATH_TO_CORPUS_ALL, top_bm25=20)
    df_train = Law_Dataset(df_train)

    # Split dataset into train and test set
    train_size = int(0.8 * len(df_train))
    test_size = len(df_train) - train_size
    train_dataset, test_dataset = random_split(
        df_train, [train_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=128, num_workers=4, shuffle=True, collate_fn=preprocessor_batch)
    test_dataloader = DataLoader(
        test_dataset, batch_size=128, num_workers=4, shuffle=False,collate_fn=preprocessor_batch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model_Paraformer().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, eps=1e-8)

    max_epochs = 20
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs}")
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            questions, articles, relevants = batch
            logits = model.forward(questions, articles)
            relevants = torch.tensor(relevants).to(device)

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

    try:
        asyncio.run(send_telegram_message(
            model_name="[TRAIN] Paraformer",
            model_parameter="",
            data_name="all_data(2022,2023,zalo)",
            alpha="none",
            top_k_bm25="20",
            accuracy=test_accuracy,
            precision=test_precision,
            recall=test_recall,
            f2=test_f2_score,
            note="question + True options"
        ))
    except Exception as e:
        print(str(e))

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F2 Score: {test_f2_score:.4f}")

    # Save the model
    torch.save(model.state_dict(), 'model_new.pth')
    print("Done !!!")


if __name__ == "__main__":
    train()
