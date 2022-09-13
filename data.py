import itertools
import random
from typing import Tuple

import numpy as np
import tensorflow as tf

from utils import iter_shuffler, rand_bool


class Dataset:
    def __init__(self, dataset_name, batch_size=512, train_proportion=0.85, validation_proportion=0.05,
                 train_pair_generation_strategy="sample",
                 validation_pair_generation_strategy="sample",
                 test_pair_generation_strategy="sample",
                 shuffler_buffer_size=4096, k_negative_samples=3, shuffle_data=False, remove_redundant=True):
        assert train_pair_generation_strategy in {"sample",
                                                  "all"}, f"Invalid pair_generation_strategy value {train_pair_generation_strategy}"
        self.name = dataset_name
        self.batch_size = batch_size
        self.train_pair_selection_strategy = train_pair_generation_strategy
        self.validation_pair_selection_strategy = validation_pair_generation_strategy
        self.test_pair_selection_strategy = test_pair_generation_strategy
        self.shuffler_buffer_size = shuffler_buffer_size
        self.k_negative_samples = k_negative_samples
        x = np.array([[int(item[1:]) for item in seq] for seq in
                      np.load(f"dataset/{dataset_name}_x_data.npy", allow_pickle=True)], dtype=object)
        y = np.load(f"dataset/{dataset_name}_y_data.npy", allow_pickle=True)
        if remove_redundant:
            data_tbl = {1 - i if dataset_name == "hadoop" else i: np.unique(x[y == i]) for i in range(2)}
        else:
            data_tbl = {1 - i if dataset_name == "hadoop" else i: x[y == i] for i in range(2)}
        if shuffle_data:
            np.random.shuffle(data_tbl[0])
            np.random.shuffle(data_tbl[1])
        self.negative_events = set([e for seq in data_tbl[0] for e in seq])
        self.positive_events = set([e for seq in data_tbl[1] for e in seq])
        self.max_event_number = max(self.positive_events.union(self.negative_events))
        self.data_table = data_tbl
        self.train_data_table = self.chunk_data_table(data_tbl, 0, train_proportion)
        self.validation_data_table = self.chunk_data_table(data_tbl, train_proportion,
                                                           train_proportion + validation_proportion)
        self.test_data_table = self.chunk_data_table(data_tbl, train_proportion + validation_proportion, 1)

    # custom test sets
    def create_aot_data_table(self, proportion):
        return {target: Dataset.truncate_sequences(sequences, proportion) for target, sequences in
                self.test_data_table.items()}

    def create_noisy_data_table(self, proportion):
        if proportion == 0:
            return self.test_data_table
        return {target: Dataset.noisify_sequences(sequences, proportion) for target, sequences in
                self.test_data_table.items()}

    # train properties
    @property
    def train_e2e_generator(self):
        while True:
            gen = self.epoch_e2e_generator(self.train_data_table, self.train_pair_selection_strategy)
            yield from self.batch_e2e_generator(gen, self.train_steps_per_epoch)

    @property
    def train_pair_generator(self):
        while True:
            gen = self.epoch_pairs_generator(self.train_data_table, self.train_pair_selection_strategy)
            yield from self.batch_pairs_generator(gen, self.train_steps_per_epoch)

    @property
    def train_steps_per_epoch(self) -> int:
        return int(np.ceil(self.train_pairs_count / self.batch_size))

    @property
    def train_negative_sample_count(self) -> int:
        return len(self.train_data_table[0])

    @property
    def train_positive_sample_count(self) -> int:
        return len(self.train_data_table[1])

    @property
    def train_pairs_count(self) -> int:
        if self.train_pair_selection_strategy == "all":
            return self.train_positive_sample_count ** 2 + self.train_negative_sample_count ** 2 \
                   + self.train_positive_sample_count * self.train_negative_sample_count \
                   - self.train_positive_sample_count - self.train_negative_sample_count
        elif self.train_pair_selection_strategy == "sample":
            return (self.train_positive_sample_count + self.train_negative_sample_count) * (
                    self.k_negative_samples + 1)

    @property
    def train_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.one_to_one(self.train_data_table)

    def deep_log_train(self, max_sequence_length) -> Tuple[np.ndarray, np.ndarray]:
        neg_samples = np.concatenate([self.train_data_table[0], self.validation_data_table[0]])
        train_x = []
        train_y = []
        tbl = {e: i for i, e in enumerate(self.negative_events, start=1)}
        for sample in neg_samples:
            # sample = [0] * (max_sequence_length - 1) + sample
            sample = [tbl[e] for e in sample]
            for subs in self.sub_sequences(sample, max_sequence_length):
                train_x.append(subs[:-1])
                train_y.append(subs[-1])
        return tf.keras.preprocessing.sequence.pad_sequences(train_x), np.array(train_y)

    def deep_log_train2(self) -> Tuple[np.ndarray, np.ndarray]:
        neg_samples = np.concatenate([self.train_data_table[0], self.validation_data_table[0]])
        train_x = []
        train_y = []
        tbl = {e: i for i, e in enumerate(self.negative_events, start=1)}
        for sample in neg_samples:
            if len(sample) < 2:
                continue
            sample = [tbl[e] for e in sample]
            train_x.append(sample[:-1])
            train_y.append(sample[1:])
        return tf.keras.preprocessing.sequence.pad_sequences(train_x), tf.keras.preprocessing.sequence.pad_sequences(
            train_y),

    # validation properties
    @property
    def validation_pairs(self):
        while True:
            gen = self.epoch_pairs_generator(self.validation_data_table, self.validation_pair_selection_strategy)
            return self.vectorize(s for s in gen)

    @property
    def validation_e2e(self):
        while True:
            gen = self.epoch_e2e_generator(self.validation_data_table, self.validation_pair_selection_strategy)
            return self.vectorize_with_three_targets(s for s in gen)

    @property
    def validation_negative_sample_count(self) -> int:
        return len(self.validation_data_table[0])

    @property
    def validation_positive_sample_count(self) -> int:
        return len(self.validation_data_table[1])

    @property
    def validation_pairs_count(self) -> int:
        return self.validation_positive_sample_count ** 2 + self.validation_negative_sample_count ** 2 \
               + self.validation_positive_sample_count * self.validation_negative_sample_count \
               - self.validation_positive_sample_count - self.validation_negative_sample_count

    @property
    def validation_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.one_to_one(self.validation_data_table)

    def deep_log_eval(self, max_sequence_length, data_type) -> Tuple[np.ndarray, np.ndarray]:
        neg_samples = np.concatenate(
            [self.test_data_table[0]]) if data_type == 'negative' else np.concatenate(
            [self.test_data_table[1], self.validation_data_table[1], self.test_data_table[1]]
        )
        train_x = []
        train_y = []
        for sample in neg_samples:
            if len(sample) < 2:
                continue
            for subs in self.sub_sequences(sample, max_sequence_length):
                train_x.append(subs[:-1])
                train_y.append(subs[-1])
        return tf.keras.preprocessing.sequence.pad_sequences(train_x), np.array(train_y)

    def deep_log_validation(self, max_sequence_length) -> Tuple[np.ndarray, np.ndarray]:
        neg_samples = self.test_data_table[0]
        train_x = []
        train_y = []
        tbl = {e: i for i, e in enumerate(self.negative_events, start=1)}
        for sample in neg_samples:
            # sample = [0] * (max_sequence_length - 1) + sample
            sample = [tbl[e] for e in sample]
            for subs in self.sub_sequences(sample, max_sequence_length):
                train_x.append(subs[:-1])
                train_y.append(subs[-1])
        return tf.keras.preprocessing.sequence.pad_sequences(train_x), np.array(train_y)

    # test properties
    @property
    def test_pairs(self):
        while True:
            gen = self.epoch_pairs_generator(self.test_data_table, self.test_pair_selection_strategy)
            return self.vectorize(s for s in gen)

    @property
    def test_negative_sample_count(self) -> int:
        return len(self.test_data_table[0])

    @property
    def test_positive_sample_count(self) -> int:
        return len(self.test_data_table[1])

    @property
    def test_pairs_count(self) -> int:
        return self.validation_positive_sample_count ** 2 + self.validation_negative_sample_count ** 2 \
               + self.validation_positive_sample_count * self.validation_negative_sample_count \
               - self.validation_positive_sample_count - self.validation_negative_sample_count

    @property
    def test_sample_count(self) -> int:
        return self.test_positive_sample_count + self.test_negative_sample_count

    @property
    def test_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.one_to_one(self.test_data_table)

    def deep_log_test(self):
        neg_samples = np.concatenate([self.test_data_table[0]])
        pos_samples = np.concatenate([self.test_data_table[1], self.validation_data_table[1], self.test_data_table[1]])
        test_x = np.concatenate([neg_samples, pos_samples])
        test_y = np.array([0] * len(neg_samples) + [1] * len(pos_samples))
        return test_x, test_y

    # data processing helpers
    @staticmethod
    def vectorize(data):
        x1, x2, y = [s for s in zip(*data)]
        x1 = tf.keras.preprocessing.sequence.pad_sequences(x1)
        x2 = tf.keras.preprocessing.sequence.pad_sequences(x2)
        y = np.array(y)
        return [x1, x2], [y]

    @staticmethod
    def vectorize_with_three_targets(data):
        x1, x2, ys, y1 = [s for s in zip(*data)]
        x1 = tf.keras.preprocessing.sequence.pad_sequences(x1)
        x2 = tf.keras.preprocessing.sequence.pad_sequences(x2)
        ys = np.array(ys)
        y1 = np.array(y1)
        return [x1, x2], [ys, y1]

    @staticmethod
    def sample_strategy_pairs_generator(data_table, k):
        for sp in data_table[0]:
            yield sp, np.random.choice(data_table[0]), 1
            for _ in range(k):
                yield sp, np.random.choice(data_table[1]), 0

        for sn in data_table[1]:
            yield sn, np.random.choice(data_table[1]), 1
            for _ in range(k):
                yield sn, np.random.choice(data_table[0]), 0

    @staticmethod
    def all_strategy_pairs_generator(data_table):
        for s1, s2 in itertools.product(data_table[0], data_table[0]):
            if s1 != s2:
                yield s1, s2, 1
        for s1, s2 in itertools.product(data_table[1], data_table[1]):
            if s1 != s2:
                yield s1, s2, 1
        for s1, s2 in itertools.product(data_table[0], data_table[1]):
            yield s1, s2, 0

    def epoch_pairs_generator(self, data_table, strategy):
        if strategy == "all":
            return iter_shuffler(self.all_strategy_pairs_generator(data_table), self.shuffler_buffer_size)
        elif strategy == "sample":
            return iter_shuffler(self.sample_strategy_pairs_generator(data_table, self.k_negative_samples),
                                 self.shuffler_buffer_size)
        else:
            raise ValueError(f"Invalid pair strategy value {self.train_pair_selection_strategy}")

    @staticmethod
    def one_to_one(data_table):
        x = tf.keras.preprocessing.sequence.pad_sequences(
            np.concatenate([data_table[0], data_table[1]]))
        y = np.concatenate(
            [np.zeros(len(data_table[0])), np.ones(len(data_table[1]))])
        return x, y

    @staticmethod
    def sample_strategy_e2e_generator(data_table, k):
        for sp in data_table[0]:
            yield sp, np.random.choice(data_table[0]), 1, 0
            for _ in range(k):
                yield sp, np.random.choice(data_table[1]), 0, 0

        for sn in data_table[1]:
            yield sn, np.random.choice(data_table[1]), 1, 1
            for _ in range(k):
                yield sn, np.random.choice(data_table[0]), 0, 1

    @staticmethod
    def all_strategy_e2e_generator(data_table):
        for s1, s2 in itertools.product(data_table[0], data_table[0]):
            if s1 != s2:
                yield s1, s2, 1, 0
        for s1, s2 in itertools.product(data_table[1], data_table[1]):
            if s1 != s2:
                yield s1, s2, 1, 1
        for s1, s2 in itertools.product(data_table[0], data_table[1]):
            yield s1, s2, 0, 0

    def epoch_e2e_generator(self, data_table, strategy):
        if strategy == "all":
            return iter_shuffler(self.all_strategy_e2e_generator(data_table), self.shuffler_buffer_size)
        elif strategy == "sample":
            return iter_shuffler(self.sample_strategy_e2e_generator(data_table, self.k_negative_samples),
                                 self.shuffler_buffer_size)
        else:
            raise ValueError(f"Invalid pair strategy value {self.train_pair_selection_strategy}")

    def batch_pairs_generator(self, gen, steps):
        for _ in range(steps):
            batch = [s for _, s in zip(range(self.batch_size), gen)]
            yield self.vectorize(batch)

    def batch_e2e_generator(self, gen, steps):
        for _ in range(steps):
            batch = [s for _, s in zip(range(self.batch_size), gen)]
            yield self.vectorize_with_three_targets(batch)

    # other helpers
    @staticmethod
    def chunk_data_table(data_table, start_prop=0.0, stop_prop=1.0):
        assert 0 <= start_prop < 1 and 0 < stop_prop <= 1, "Invalid proportion value"
        return {target: sequences[int(start_prop * len(sequences)):int(stop_prop * len(sequences))] for
                target, sequences in
                data_table.items()}

    @staticmethod
    def truncate_sequences(sequences, proportion: float, min_len=None):
        assert 0 < proportion <= 1, "proportion is invalid"
        if min_len is None:
            min_len = np.ceil(proportion ** -1)
        return np.array(
            [seq[:int(np.ceil(len(seq) * proportion))] if len(sequences) > min_len else seq for seq in sequences])

    @staticmethod
    def noisify_sequences(sequences, proportion: float, duplicate_proportion=0.3, remove_proportion=0.3,
                          shuffle_proportion=0.3):
        seqs = []
        for seq in sequences:
            if rand_bool(proportion):
                seq = np.copy(seq)
                modification_types = ["duplicate", "remove", "shuffle"]
                if len(seq) <= remove_proportion ** -1:
                    modification_types.remove("remove")
                if len(seq) <= (shuffle_proportion ** -1) * 2:
                    modification_types.remove("shuffle")
                noise_type = random.choice(modification_types)
                if noise_type == "duplicate":
                    for _ in range(int(duplicate_proportion * len(seq))):
                        index = np.random.randint(0, len(seq))
                        event = seq[index]
                        seq = np.insert(seq, index, event)
                elif noise_type == "remove":
                    for _ in range(int(remove_proportion * len(seq))):
                        index = np.random.randint(0, len(seq))
                        seq = np.delete(seq, index)
                elif noise_type == "shuffle":
                    shuffle_len = int(shuffle_proportion * len(seq))
                    index = np.random.randint(0, len(seq) - shuffle_len)
                    shuffle_seq = seq[index:index + shuffle_len]
                    np.random.shuffle(shuffle_seq)
                    seq[index:index + shuffle_len] = shuffle_seq
            seqs.append(seq)
        return np.array(seqs, dtype=np.object)

    @staticmethod
    def sub_sequences(sequence, length):
        if len(sequence) < length:
            sequence = ([0] * (length - len(sequence))) + sequence
        if len(sequence) == length:
            yield sequence
        for i in range(len(sequence) - length):
            yield sequence[i:i + length]
