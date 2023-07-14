import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from my_attention import Attention


class Model_Paraformer(nn.Module):
    def __init__(self, base_model="keepitreal/vietnamese-sbert"):
        super(Model_Paraformer, self).__init__()
        self.sentenceTransformer = SentenceTransformer(base_model)
        self.attention = Attention(768)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)
        # self.sigmod = nn.Sigmoid()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query, article):
        query_vector = self.sentenceTransformer.encode(
            query, convert_to_tensor=True)

        '''>>> torch.Size([1, 768]) -> torch.Size([1, 1, 768])'''
        query_vector = torch.unsqueeze(query_vector, 0)

        article_vector = torch.stack([self.sentenceTransformer.encode(
            sentence, convert_to_tensor=True) for sentence in article])

        '''>>> torch.Size([number_sentence, 1, 768]) -> torch.Size([1, number_sentence, 768]) '''
        article_vector = article_vector.permute(1, 0, 2)

        attention, _ = self.attention(query_vector, article_vector)

        output = self.classifier(attention)

        '''>>> torch.Size([1, 1, 2]) -> torch.Size([1, 2])'''
        output = torch.squeeze(output, 1)
        # output = self.sigmod(output)
        return output

    def predict(self, query, article):
        with torch.no_grad():
            query_vector = self.sentenceTransformer.encode(
                query, convert_to_tensor=True)

            '''>>> torch.Size([1, 768]) -> torch.Size([1, 1, 768])'''
            query_vector = torch.unsqueeze(query_vector, 1)
            article_vector = torch.stack([self.sentenceTransformer.encode(
                sentence, convert_to_tensor=True) for sentence in article])

            '''>>> torch.Size([number_sentence, 1, 768]) -> torch.Size([1, number_sentence, 768]) '''
            article_vector = article_vector.permute(1, 0, 2)

            attention, _ = self.attention(query_vector, article_vector)

            output = self.classifier(attention)

            '''>>> torch.Size([1, 1, 2]) -> torch.Size([1, 2])'''
            output = torch.squeeze(output, 1)
            '''>>> torch(1) -> torch([1])'''
            return torch.unsqueeze(torch.argmax(output).cpu().detach(), 0)

    def get_score(self, query, article):
        with torch.no_grad():
            query_vector = self.sentenceTransformer.encode(
                query, convert_to_tensor=True)

            '''>>> torch.Size([768]) -> torch.Size([1, 1, 768])'''
            query_vector = torch.unsqueeze(query_vector, 0)
            query_vector = torch.unsqueeze(query_vector, 0)

            article_vector = torch.stack([self.sentenceTransformer.encode(
                sentence, convert_to_tensor=True) for sentence in article])
            article_vector = torch.unsqueeze(article_vector, 1)
            article_vector = article_vector.permute(1, 0, 2)

            attention, _ = self.attention(query_vector, article_vector)

            output = self.classifier(attention)

            output = torch.squeeze(output, 1)
            return output.detach().numpy()[0][1]
