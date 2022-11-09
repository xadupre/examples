import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset, TensorDataset


class CustomDataset(Dataset):
    def __init__(self, WORLD_SIZE, encoded_tensors, labels, device, local_rank=-1):
        if local_rank == -1:
            self.encoded_tensors = [e.to(device) for e in encoded_tensors]
            self.labels = labels.to(torch.float32).to(device)
        else:
            self.encoded_tensors = [e.to(device) for e in encoded_tensors]
            self.labels = labels.to(torch.float32).to(device)[local_rank::WORLD_SIZE]
        self.transform = None
        self.target_transform = None
        self.local_rank = local_rank

    def __len__(self):
        return len(self.encoded_tensors)

    def __getitem__(self, idx):
        return self.encoded_tensors[idx], self.labels[idx]


def load_dataset_train(path="data/wiki.train.tokens"):
    # unused
    # see https://github.com/knagrecha/hydra/blob/main/examples/utils.py
    raw_text = ""
    with open(path, "r") as fp:
        raw_text += fp.read()
    raw_text += "<|endoftext|>"
    tokens = np.stack(tokenizer.encode(raw_text))
    return tokens


def get_data_loader_train(batch_size, context_length=512):
    # unused
    data = lazy_load_train()[0]
    # Chunk data by context_length
    ds = Subset(
        data,
        [
            slice(i, i + context_length)
            for i in range(0, len(data) - (len(data) % context_length), context_length)
        ],
    )
    data_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    return data_loader
