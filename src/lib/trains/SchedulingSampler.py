from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import re
from torch.utils.data.sampler import Sampler, SequentialSampler


class SchedulingSampler(Sampler):

    def __init__(self, dataset, batch_size: int, drop_last: bool, epoch_init:bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.dataset = dataset
        self.sampler = SequentialSampler(dataset)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.cds_seq_start = dataset.cds_seq
        self.cds_seq_end = [x-1 for x in self.cds_seq_start[1:]]
        self.cds_seq_end.append(dataset.nds[-1]-1)
        self.seq_len = dataset.nds_seq
        self.seq_inds = list(range(len(self.seq_len)))
        self.seq_inds_vector = list(range(self.batch_size))
        self.epoch_init = epoch_init

        self.start_iter = [-1]



    def __iter__(self):
        batch = []
        start_idx = 0

        if not self.epoch_init:
            self.start_iter[0] += 1
            start_idx = self.start_iter[0] * ((len(self.sampler) + self.batch_size - 1) // self.batch_size)

        for idx in range(len(self.sampler)):
            for seq_idx in self.seq_inds_vector:
                _idx = idx + self.cds_seq_start[seq_idx] + start_idx
                while _idx > self.cds_seq_end[seq_idx]:
                    _idx = _idx -  self.seq_len[seq_idx]
                batch.append(_idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []


        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        # return max(self.seq_len)
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size