import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from logging import getLogger

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.envs.environment import do_stats

logger = getLogger()


def detokenize(data, args, env, executor=None):
    res = []
    pars = env.tokenizer.dataclass._save_class_params()
    if args.process_pool:
        BATCH = args.gen_batch_size
        data_slices = [data[i : i + BATCH] for i in range(0, len(data), BATCH)]

        if executor is not None:
            for chunk in executor.map(env.tokenizer.decode_batch, data_slices, repeat(pars, len(data_slices))):
                if chunk:
                    res.extend(chunk)
        else:
            with ProcessPoolExecutor(max_workers=min(20, args.num_workers)) as ex:
                for chunk in ex.map(env.tokenizer.decode_batch, data_slices, repeat(pars, len(data_slices))):
                    if chunk:
                        res.extend(chunk)
    else:
        res = env.tokenizer.decode_batch(data, pars)
    return res


# helper functions for creating the training and test Datasets


def generate_and_score(args, classname):
    """
    Generation method if no data
    """
    data = []
    BATCH = args.gen_batch_size
    batch_counts = [BATCH] * (args.gensize // BATCH)
    rem = args.gensize % BATCH
    if rem:
        batch_counts.append(rem)
    with tqdm(total=args.gensize, desc="Generating data", unit="ex") as pbar:
        if args.process_pool:
            pars = classname._save_class_params()
            with ProcessPoolExecutor(max_workers=min(20, args.num_workers)) as executor:
                # map returns lists; stream them to avoid a giant materialization
                for chunk in executor.map(
                    classname._batch_generate_and_score, batch_counts, repeat(args.N, len(batch_counts)), repeat(pars, len(batch_counts))
                ):
                    if chunk:  # extend incrementally to manage memory
                        data.extend(chunk)
                        pbar.update(len(chunk))
        else:
            for t in batch_counts:
                d = classname._batch_generate_and_score(t, args.N)
                if d is not None:
                    data.extend(d)
                    pbar.update(len(d))
    return data


def select_best(n, data):
    if len(data) <= n:
        random.shuffle(data)
        return data
    sorted_data = sorted(data, key=lambda x: x.score, reverse=True)[:n]
    random.shuffle(sorted_data)
    return sorted_data


def make_train_test(data, ntest):
    """
    Create a train and test dataset from a dataset.
    """
    indices = np.random.permutation(len(data))
    rp = [data[i] for i in indices]
    return rp[:-ntest], rp[-ntest:]


def compute_unique_data(old_data, new_data=None):
    def add_unique(src, unique_hashes):
        des = []
        for d in src:
            if d.features not in unique_hashes:
                unique_hashes.add(d.features)
                des.append(d)
        return des, unique_hashes

    unique_hashes = set()
    unique_old_data, unique_hashes = add_unique(old_data, unique_hashes)
    if new_data is not None:
        unique_new_data, unique_hashes = add_unique(new_data, unique_hashes)
    else:
        unique_new_data = None
    return unique_old_data, unique_new_data


def update_datasets(args, data, train_set, test_set, train_path, test_path):
    inc_temp = False
    if args.keep_only_unique:
        bef = len(data)
        data, _ = compute_unique_data(data)
        aft = len(data)
        logger.info(f"Unique processing: {aft} examples left, {bef-aft} duplicates")
        do_stats(-1, data)
        if aft / (bef + 1) < 0.9:
            inc_temp = True
    if args.new_proportion > 0.0:
        new_data = select_best(int(args.new_proportion * args.pop_size), data)
    else:
        new_data = select_best(args.pop_size, data)

    if len(new_data) >= 2 * args.ntest or test_set is None:
        new_train, test_set = make_train_test(new_data, args.ntest)
    else:
        new_train = new_data
    logger.info(f"New train and test generated. Size are train: {len(new_train)}, test {len(test_set)}")
    # Get all examples of previous train and current train and then select best.
    if args.keep_only_unique:
        train_set, new_train = compute_unique_data(train_set, new_train)
        logger.info(f"Unique data computed for original train set: {len(train_set)}, generated train set: {len(new_train)}")
    if args.new_proportion > 0.0:
        train_set = select_best(int((1.0 - args.new_proportion) * args.pop_size), train_set) + new_train
    else:
        train_set = select_best(args.pop_size, train_set + new_train)
    logger.info(f"Final train and test generated. Size are train: {len(train_set)}, test {len(test_set)}")

    pickle.dump(test_set, open(test_path, "wb"))
    pickle.dump(train_set, open(train_path, "wb"))
    return train_set, test_set, inc_temp


def load_initial_data(args, classname):
    train_data_path = os.path.join(args.dump_path, "train_data.pkl")
    test_data_path = os.path.join(args.dump_path, "test_data.pkl")
    if os.path.isfile(train_data_path):
        logger.info("resuming from existing data")
        train_set = pickle.load(open(train_data_path, "rb"))
        test_set = pickle.load(open(test_data_path, "rb"))
    else:
        data = generate_and_score(args, classname=classname)
        test_set = []
        train_set = []
        train_set, test_set, _ = update_datasets(args, data, train_set, test_set, train_data_path, test_data_path)
    return train_set, test_set


class CharDataset(Dataset):
    def __init__(self, encoded_data, max_len, stoi):
        self.encoded_data = encoded_data
        self.max_len = max_len
        self.pad_token_id = stoi["PAD"]

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

    def collate_fn(self, batch):
        x = np.full((len(batch), self.max_len + 2), self.pad_token_id, dtype=np.int32)

        for i, el in enumerate(batch):
            x[i, : el.shape[0]] = el
        valid_col = (x != self.pad_token_id).any(axis=0)
        last_col = np.nonzero(valid_col)[0][-1] + 1
        x = x[:, :last_col]
        y = np.concatenate([x[:, 1:], np.full((len(batch), 1), self.pad_token_id, dtype=x.dtype)], axis=1)
        return torch.LongTensor(x), torch.LongTensor(y)


class InfiniteDataLoader:
    """
    Create a infinite datalaoder in PyTorch
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, collate_fn=dataset.collate_fn, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:  # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch
