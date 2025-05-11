import pathlib
import json
import numpy as np
import tqdm
import imblearn
import sklearn
import shutil
from enum import Enum

class BatchMode(Enum):
    FULL = 0
    TRAIN_TEST = 1
    SHOWCASE = 2

class BatchManager:
    def __init__(self, directory):
        self.batch_suffix = '.npy'
        self.batch_info_suffix = '.txt'

        self.dirs = {'main': pathlib.Path(directory)}
        for subdir in ['X_train', 'X_test', 'y_train', 'y_test', 'intact', 'coords']:
            self.dirs[subdir] = self.dirs['main'] / subdir

    def dump_singleton_batch(self, subdir, batch):
        self.dirs[subdir].mkdir(parents=True, exist_ok=True)
        np.save(self.dirs[subdir] / f'singleton{self.batch_suffix}', batch)

    def load_singleton_batch(self, subdir):
        return np.load(self.dirs[subdir] / f'singleton{self.batch_suffix}')

    def dump_batches(self, subdir, entry_id, batches, bpi):
        self.dirs[subdir].mkdir(parents=True, exist_ok=True)

        for batch_id, batch in enumerate(np.array_split(batches, bpi)):
            np.save(self.dirs[subdir] / f'{entry_id}_{batch_id}.npy', batch)

    def load_batch(self, subdir, entry_id, batch_id):
        return np.load(self.dirs[subdir] / f'{entry_id}_{batch_id}.npy')

    def dump_batch_info(self, batch_info, batch_mode):
        batch_info_path = self.dirs['main'] / f'batch_info_{batch_mode.name.lower()}{self.batch_info_suffix}'

        with open(batch_info_path, 'w') as f:
            f.write(batch_info.to_json(batch_mode))

    def load_batch_info(self, batch_mode):
        batch_info_path = self.dirs['main'] / f'batch_info_{batch_mode.name.lower()}{self.batch_info_suffix}'

        try:
            with open(batch_info_path, 'r') as f:
                return BatchInfo.from_json(f.read())
        except FileNotFoundError:
            return None

    def remove_dirs(self, dirs):
        for dir in dirs:
            if self.dirs[dir].exists():
                shutil.rmtree(self.dirs[dir])

class BatchInfo:
    def __init__(self, **kwargs):
        self.image_channels =        kwargs.pop('image_channels', [1])         # int list
        self.patch_side =            kwargs.pop('patch_side', None)            # int
        self.patch_step_train_test = kwargs.pop('patch_step_train_test', None) # int
        self.patch_step_showcase =   kwargs.pop('patch_step_showcase', None)   # int
        self.entry_ids_train_test =  kwargs.pop('entry_ids_train_test', None)  # int list
        self.entry_ids_showcase =    kwargs.pop('entry_ids_showcase', None)    # int list
        self.bpi_test =              kwargs.pop('bpi_test', None)              # int
        self.bpi_showcase =          kwargs.pop('bpi_showcase', None)          # int
        self.train_size =            kwargs.pop('train_size', None)            # float
        self.under_sampling_strat =  kwargs.pop('under_sampling_strat', None)  # float
        self.batch_mode =            BatchMode(kwargs.pop('batch_mode', 0))    # BatchMode

    def to_json(self, batch_mode):
        if batch_mode == BatchMode.FULL:
            return json.dumps({
                'image_channels':        self.image_channels,
                'patch_side':            self.patch_side,
                'patch_step_train_test': self.patch_step_train_test,
                'patch_step_showcase':   self.patch_step_showcase,
                'entry_ids_train_test':  self.entry_ids_train_test,
                'entry_ids_showcase':    self.entry_ids_showcase,
                'bpi_test':              self.bpi_test,
                'bpi_showcase':          self.bpi_showcase,
                'train_size':            self.train_size,
                'under_sampling_strat':  self.under_sampling_strat,
                'batch_mode':            batch_mode.value
            })
        elif batch_mode == BatchMode.TRAIN_TEST:
            return json.dumps({
                'image_channels':        self.image_channels,
                'patch_side':            self.patch_side,
                'patch_step_train_test': self.patch_step_train_test,
                'entry_ids_train_test':  self.entry_ids_train_test,
                'bpi_test':              self.bpi_test,
                'train_size':            self.train_size,
                'under_sampling_strat':  self.under_sampling_strat,
                'batch_mode':            batch_mode.value
            })
        elif batch_mode == BatchMode.SHOWCASE:
            return json.dumps({
                'image_channels':        self.image_channels,
                'patch_side':            self.patch_side,
                'patch_step_showcase':   self.patch_step_showcase,
                'entry_ids_showcase':    self.entry_ids_showcase,
                'bpi_showcase':          self.bpi_showcase,
                'batch_mode':            batch_mode.value
            })

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(**data)

    def __eq__(self, other):
        compared_batch_mode = BatchMode(max(self.batch_mode.value, other.batch_mode.value))
        return self.to_json(compared_batch_mode) == other.to_json(compared_batch_mode)

def create_patches(dataset_manager, batch_info, entry_id, batch_mode):
    image = dataset_manager.load('image', entry_id)[:, :, tuple(batch_info.image_channels)]
    gold = dataset_manager.load('gold', entry_id)

    patch_step = None
    if batch_mode == BatchMode.TRAIN_TEST:
        patch_step = batch_info.patch_step_train_test
    elif batch_mode == BatchMode.SHOWCASE:
        patch_step = batch_info.patch_step_showcase

    rows = 1 + (dataset_manager.height - batch_info.patch_side) // patch_step
    columns = 1 + (dataset_manager.width - batch_info.patch_side) // patch_step
    offset = batch_info.patch_side // 2

    number_of_patches = rows * columns
    patch_size = batch_info.patch_side ** 2 * len(batch_info.image_channels)

    patches = np.zeros((number_of_patches, patch_size), dtype=np.float32)
    labels = None
    coords = None
    if batch_mode == BatchMode.TRAIN_TEST:
        labels = np.zeros((number_of_patches), dtype=bool)
    elif batch_mode == BatchMode.SHOWCASE:
        coords = np.zeros((number_of_patches, 2), dtype=np.int32)

    for i in range(rows):
        for j in range(columns):
            y = offset + i * patch_step
            x = offset + j * patch_step

            y_begin, y_end = y - offset, y + offset + 1
            x_begin, x_end = x - offset, x + offset + 1

            square_patch = image[y_begin:y_end, x_begin:x_end]

            index = i * columns + j
            patches[index] = square_patch.flatten()
            if batch_mode == BatchMode.TRAIN_TEST:
                labels[index] = gold[y, x]
            elif batch_mode == BatchMode.SHOWCASE:
                coords[index] = (y, x)

    if batch_mode == BatchMode.TRAIN_TEST:
        return patches, labels
    elif batch_mode == BatchMode.SHOWCASE:
        return patches, coords
    return patches

def batch(dataset_manager, batch_manager, batch_info):
    batch_info_train_test = batch_manager.load_batch_info(BatchMode.TRAIN_TEST)

    if batch_info_train_test is not None and batch_info_train_test == batch_info:
        print('Batch request \'train_test\' already satisfied')
    else:
        batch_manager.remove_dirs(['X_train', 'X_test', 'y_train', 'y_test'])

        X_train = []
        y_train = []

        for entry_id in tqdm.tqdm(batch_info.entry_ids_train_test, desc='Batching train_test', unit='entry'):
            patches, labels = create_patches(dataset_manager, batch_info, entry_id, BatchMode.TRAIN_TEST)

            X_train_part, X_test, y_train_part, y_test = sklearn.model_selection.train_test_split(
                patches, labels, train_size=batch_info.train_size
            )

            if batch_info.under_sampling_strat is not None:
                sampler = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=batch_info.under_sampling_strat)
                X_train_part, y_train_part = sampler.fit_resample(X_train_part, y_train_part)

            X_train.append(X_train_part)
            y_train.append(y_train_part)
            batch_manager.dump_batches('X_test', entry_id, X_test, batch_info.bpi_test)
            batch_manager.dump_batches('y_test', entry_id, y_test, batch_info.bpi_test)

        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        batch_manager.dump_singleton_batch('X_train', X_train)
        batch_manager.dump_singleton_batch('y_train', y_train)

        batch_manager.dump_batch_info(batch_info, BatchMode.TRAIN_TEST)

    batch_info_showcase = batch_manager.load_batch_info(BatchMode.SHOWCASE)

    if batch_info_showcase is not None and batch_info_showcase == batch_info:
        print('Batch request \'showcase\' already satisfied')
    else:
        batch_manager.remove_dirs(['intact', 'coords'])

        for entry_id in tqdm.tqdm(batch_info.entry_ids_showcase, desc='Batching showcase', unit='entry'):
            patches, coords = create_patches(dataset_manager, batch_info, entry_id, BatchMode.SHOWCASE)

            batch_manager.dump_batches('intact', entry_id, patches, batch_info.bpi_showcase)
            batch_manager.dump_batches('coords', entry_id, coords, batch_info.bpi_showcase)

        batch_manager.dump_batch_info(batch_info, BatchMode.SHOWCASE)
