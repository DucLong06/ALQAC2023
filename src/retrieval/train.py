import argparse
import datetime
import os
import re
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from src.utils.bot_telegram import send_message, send_telegram_message
from src.utils.early_stopping import EarlyStopping
from src.utils.eval_metrics import eval_model
from src.retrieval.law_data import Law_Dataset
from src.retrieval.model_paraformer import Model_Paraformer
from src.retrieval.raw_data import data_training_generator
import src.utils.my_env as my_env
import asyncio
from tqdm import tqdm
import src.utils.my_logger as my_logger


logger = my_logger.Logger("training", my_env.LOG)


def preprocessor_batch(batch):
    questions, articles, relevants = zip(*batch)

    max_article_length = max(len(article) for article in articles)

    padded_articles = [article + [''] * (max_article_length - len(article)) if len(
        article) < max_article_length else article for article in articles]

    return questions, padded_articles, relevants


def generate_model_name(original_string, path_to_save=my_env.PATH_TO_SAVE_MODEL):
    now = datetime.datetime.now()
    time_string = now.strftime("%Y%m%d%H%M%S")
    clean_time_string = re.sub(r'\W+', '', time_string)
    original_string = original_string.replace("/", "-")
    return os.path.join(path_to_save,
                        f"F_{original_string}_{clean_time_string}.pth")


def train(base_model, input_questions, input_articles, top_bm25, batch_size, max_epochs):
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

    model = Model_Paraformer(base_model).to(device)

    optimizer = model.configure_optimizers()

    path_name_model = generate_model_name(base_model)
    early_stopping = EarlyStopping(patience=3, verbose=True, delta=0.0001)

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

            # Validation loss calculation without backpropagation
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

    # Evaluate the model on the test set

    _, test_accuracy, test_precision, test_recall, test_f2_score = eval_model(
        test_dataloader, model)

    try:
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
    except Exception as e:
        logger.error(str(e))

    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test Precision: {test_precision:.4f}")
    logger.info(f"Test Recall: {test_recall:.4f}")
    logger.info(f"Test F2 Score: {test_f2_score:.4f}")

    # Save the model
    torch.save(model.state_dict(), path_name_model)
    logger.info("Done Training !!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str,
                        default="keepitreal/vietnamese-sbert",
                        help="Base model to use for Paraformer.")
    parser.add_argument('--input_questions', type=str,
                        default=my_env.PATH_TO_PUBLIC_TRAIN,
                        help="Path to the input questions data file.")
    parser.add_argument('--input_articles', type=str,
                        default=my_env.PATH_TO_CORPUS_2023,
                        help="Path to the input articles data file.")
    parser.add_argument('--top_bm25', type=int, default=10,
                        help="Number of top fake BM25 articles to consider.")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size for training.")
    parser.add_argument('--max_epochs', type=int, default=20,
                        help="Maximum number of epochs for training.")
    args = parser.parse_args()

    train(args.base_model, args.input_questions, args.input_articles,
          args.top_bm25, args.batch_size, args.max_epochs)
