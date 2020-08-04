import numpy as np

class DataLoader:
    def __init__(self, x, y, batch_size=64):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.n_batch = len(x) // batch_size
        
    def get_item(self, idx):
        return self.x[idx], self.y[idx]
        
    def get_batch(self, idxs):
        x = []
        y = []
        for idx in idxs:
            temp_x, temp_y = self.get_item(idx)
            x.append(temp_x)
            y.append(temp_y)
        return np.array(x), np.array(y)
        
    def get_batch_idxs(self):
        idx = list(range(len(self.y)))
        np.random.shuffle(idx)
        batch_idxs = []
        for batch in range(self.n_batch):
            batch_idxs.append(idx[batch*self.batch_size:batch*self.batch_size + self.batch_size])
        return batch_idxs

    def __iter__(self):
        batch_idxs = self.get_batch_idxs()
        for batch_idx in batch_idxs:
            yield self.get_batch(batch_idx)

    def __len__(self):
        return self.n_batch