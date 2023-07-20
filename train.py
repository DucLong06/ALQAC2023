import argparse
import datetime
import os
import re
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from bot_telegram import send_message, send_telegram_message
from eval_metrics import eval_model
from law_data import Law_Dataset
from model_paraformer import Model_Paraformer
from raw_data import df_create_data_training
import my_env
import asyncio
from tqdm import tqdm
import my_logger

logger = my_logger.Logger("training", my_env.LOG)

def preprocessor_batch(batch):
    questions, articles, relevants = zip(*batch)

    max_article_length = max(len(article) for article in articles)

    padded_articles = [article + [''] * (max_article_length - len(article)) if len(
        article) < max_article_length else article for article in articles]

    return questions, padded_articles, relevants

def train_valid_test_split(dataset, train_ratio, valid_ratio, test_ratio):
    assert train_ratio + valid_ratio + test_ratio == 1, "The ratios should add up to 1."
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    valid_size = int(valid_ratio * total_size)
    test_size = total_size - train_size - valid_size

    # Use random_split to split the dataset
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    return train_dataset, valid_dataset, test_dataset

def generate_model_name(original_string):
    now = datetime.datetime.now()
    time_string = now.strftime("%Y%m%d%H%M%S")
    clean_time_string = re.sub(r'\W+', '', time_string)
    original_string = original_string.replace("/","-")
    return f"{original_string}_{clean_time_string}.pth"


def train(base_model:str="keepitreal/vietnamese-sbert"):
    df_train = df_create_data_training(
        my_env.PATH_TO_PUBLIC_TRAIN, my_env.PATH_TO_CORPUS_2023, top_bm25=20)
    df_train = Law_Dataset(df_train)

    # Split dataset into train and test set
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1
    train_dataset, valid_dataset, test_dataset = train_valid_test_split(df_train, train_ratio, valid_ratio, test_ratio)

    train_dataloader = DataLoader(
        train_dataset, batch_size=64, num_workers=4, shuffle=True, collate_fn=preprocessor_batch)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=64, num_workers=4, shuffle=False, collate_fn=preprocessor_batch)
    test_dataloader = DataLoader(
        test_dataset, batch_size=64, num_workers=4, shuffle=False,collate_fn=preprocessor_batch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model_Paraformer(base_model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, eps=1e-8)

    max_epochs = 1
    name_model = generate_model_name(base_model)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    for epoch in range(max_epochs):
        logger.info(f"Epoch {epoch+1}/{max_epochs}")
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
            
        _,accuracy, _, _, _= eval_model(valid_dataloader, model)
        train_loss = total_loss / len(train_dataloader)
       
        logger.info(f"Validation acc: {accuracy:.4f}")
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info("================================================")

    logger.info('Training finished.')

    # Evaluate the model on the test set

    _, test_accuracy, test_precision, test_recall, test_f2_score = eval_model(
        test_dataloader, model)
   
    try:
        asyncio.run(send_telegram_message(
            model_name="[TRAIN] Paraformer",
            model_base=f"base_model: {base_model}",
            data_name="train 2023",
            alpha="none",
            top_k_bm25="20",
            accuracy=test_accuracy,
            precision=test_precision,
            recall=test_recall,
            f2=test_f2_score,
            note="question + True options",
        ))
    except Exception as e:
        logger.error(str(e))

    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test Precision: {test_precision:.4f}")
    logger.info(f"Test Recall: {test_recall:.4f}")
    logger.info(f"Test F2 Score: {test_f2_score:.4f}")

    # Save the model
    torch.save(model.state_dict(), os.path.join(my_env.PATH_TO_SAVE_MODEL, name_model))
    logger.info("Done !!!")


if __name__ == "__main__":
    # for base_model in my_env.list_base_model:
    #     logger.info(f"Training: {base_model}")
    #     try:
    #         train(base_model)
    #         logger.info(f"Done: {base_model}")
    #     except Exception as e:
    #         logger.error(f"{base_model}:{str(e)}")
    #         asyncio.run(send_message(f"base model error : {base_model}"))
    train()

 