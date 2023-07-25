import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse
import datetime
import os
import re
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from bot_telegram import send_message, send_telegram_message
from early_stopping import EarlyStopping
from eval_metrics import calculate_accuracy, calculate_f2_score, calculate_precision, calculate_recall, eval_model
from src.task1.law_data import Law_Dataset
from src.task1.model_paraformer import Model_Paraformer
from model_roberta import Model_Roberta
from src.task1.raw_data import data_training_generator
import my_env
import asyncio
from tqdm import tqdm
import my_logger
from src.task1.train import generate_model_name


logger = my_logger.Logger("training", my_env.LOG)


def preprocessor_batch(batch):
    questions, articles, relevants = zip(*batch)

    max_article_length = max(len(article) for article in articles)

    padded_articles = [article + [''] * (max_article_length - len(article)) if len(
        article) < max_article_length else article for article in articles]

    return questions, padded_articles, relevants


def train_model(input_questions, input_articles, top_bm25=10, batch_size=1, max_epochs=5):

    train_df, val_df, test_df = data_training_generator(
        input_questions, input_articles, top_bm25=top_bm25, train_ratio=0.8, val_ratio=0.1)
    train_dataset = Law_Dataset(train_df)
    valid_dataset = Law_Dataset(val_df)
    test_dataset = Law_Dataset(test_df)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=preprocessor_batch)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=preprocessor_batch)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=preprocessor_batch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model_Roberta()

    optimizer = Adam(model.parameters(), lr=3e-5, eps=1e-8)
    base_model = "roberta"
    path_name_model = generate_model_name(base_model)
    early_stopping = EarlyStopping(patience=3, verbose=True, delta=0.0001)

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

        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for val_batch in tqdm(valid_dataloader):
                val_questions, val_articles, val_relevants = val_batch
                val_logits = model.forward(val_questions, val_articles)
                val_relevants = torch.tensor(val_relevants).to(device)
                val_loss = model.criterion(val_logits, val_relevants)
                val_total_loss += val_loss.item()

        avg_val_loss = val_total_loss / len(valid_dataloader)
        train_loss = total_loss / len(train_dataloader)

        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"Train Loss: {train_loss:.4f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
        logger.info("================================================")

    logger.info('Training finished.')

    try:
        print("eval model")
        model.eval()
        correct = 0
        total = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0
        total_loss = 0.0
        with torch.no_grad():
            for query, article, label in tqdm(test_dataloader):
                label = torch.tensor(label).cpu()
                output = model.predict(query, article)
                # total_loss += model.criterion(output, label).item()

                total += label.size(0)
                correct += output.eq(label).sum().item()
                true_positive += (output.eq(1) & label.eq(1)).sum().item()
                # true_negative += (output.eq(0) & label.eq(0)).sum().item()
                false_positive += (output.eq(1) & label.eq(0)).sum().item()
                false_negative += (output.eq(0) & label.eq(1)).sum().item()

        loss = total_loss/len(test_dataloader)
        test_accuracy = calculate_accuracy(correct, total)
        test_precision = calculate_precision(true_positive, false_positive)
        test_recall = calculate_recall(true_positive, false_negative)
        test_f2_score = calculate_f2_score(test_precision, test_recall)
        asyncio.run(send_telegram_message(
            model_name=f"[TRAIN]{path_name_model}",
            model_base=f"base_model: {base_model}",
            data_name=f"question:{input_questions} articles:{input_articles}",
            alpha="none",
            top_k_bm25=top_bm25,
            accuracy=test_accuracy,
            precision=test_precision,
            recall=test_recall,
            f2=test_f2_score,
            note="question + True options",
        ))
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        logger.info(f"Test F2 Score: {test_f2_score:.4f}")
    except Exception as e:
        logger.error(str(e))

    # Save the model
    torch.save(model.state_dict(), path_name_model)
    logger.info("Done Training !!!")


train_model(my_env.PATH_TO_PUBLIC_TRAIN, my_env.PATH_TO_CORPUS_2023)
