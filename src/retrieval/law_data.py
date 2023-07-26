from torch.utils.data import  Dataset


class Law_Dataset(Dataset):
    def __init__(self, df) -> None:
        self.question = df["question"]
        self.article = df["article"]
        self.relevant = df["relevant"]

    def __getitem__(self, index) -> tuple:
        return self.question[index], self.article[index], self.relevant[index]

    def __len__(self) -> int:
        return len(self.question)


