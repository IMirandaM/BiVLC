# CODE PROVIDED BY SUGARCREPE

# MIT License

# Copyright (c) 2023 RAIVN Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import numpy as np
import pandas as pd
import csv
from tqdm import trange
import argparse
from pathlib import Path


def adversarial_refine(model1_score_gap, model2_score_gap):
    n_grids = 50
    keep = []  # list of indices to keep

    score_to_idx = {}
    for i in range(1, n_grids + 1):
        for j in range(1, n_grids + 1):
            score_to_idx[(i, j)] = []
            score_to_idx[(-i, -j)] = []
            score_to_idx[(i, -j)] = []
            score_to_idx[(-i, j)] = []

    model1_zero_idx = []
    model2_zero_idx = []
    for i, (model1_gap, model2_gap) in enumerate(zip(model1_score_gap, model2_score_gap)):
        if model1_gap == 0 and model2_gap == 0:
            keep.append(i)
            continue
        if model1_gap == 0:
            model1_zero_idx.append(i)
            continue
        if model2_gap == 0:
            model2_zero_idx.append(i)
            continue

        model1_id = int(abs(model1_gap) * n_grids) + 1
        model2_id = int(abs(model2_gap) * n_grids) + 1
        if model1_gap > 0 and model2_gap > 0:
            score_to_idx[(model1_id, model2_id)].append(i)
        elif model1_gap < 0 and model2_gap < 0:
            score_to_idx[(-model1_id, -model2_id)].append(i)
        elif model1_gap > 0 and model2_gap < 0:
            score_to_idx[(model1_id, -model2_id)].append(i)
        elif model1_gap < 0 and model2_gap > 0:
            score_to_idx[(-model1_id, model2_id)].append(i)
        else:
            raise ValueError
    
    for i in range(1, n_grids + 1):
        for j in range(1, n_grids + 1):
            idx = score_to_idx[(i, j)]
            op_idx = score_to_idx[(-i, -j)]
            if len(idx) > 0 and len(op_idx) > 0:
                if len(idx) > len(op_idx):
                    keep.extend(op_idx)
                    keep.extend(list(np.random.choice(idx, len(op_idx), replace=False)))
                else:
                    keep.extend(idx)
                    keep.extend(list(np.random.choice(op_idx, len(idx), replace=False)))

            idx = score_to_idx[(i, -j)]
            op_idx = score_to_idx[(-i, j)]
            if len(idx) > 0 and len(op_idx) > 0:
                if len(idx) > len(op_idx):
                    keep.extend(op_idx)
                    keep.extend(list(np.random.choice(idx, len(op_idx), replace=False)))
                else:
                    keep.extend(idx)
                    keep.extend(list(np.random.choice(op_idx, len(idx), replace=False)))

    n_grids = 10

    if len(model1_zero_idx) > 0:
        pos_score_to_idx, neg_score_to_idx = {i: [] for i in range(n_grids)}, {i: [] for i in range(n_grids)}
        for i in model1_zero_idx:
            gap = model2_score_gap[i]
            if gap > 0:
                pos_score_to_idx[int(gap * n_grids)].append(i)
            else:
                neg_score_to_idx[int(-gap * n_grids)].append(i)
        for i in trange(n_grids):
            pos_idx = pos_score_to_idx[i]
            neg_idx = neg_score_to_idx[i]
            if len(pos_idx) > len(neg_idx):
                keep.extend(neg_idx)
                keep.extend(list(np.random.choice(pos_idx, len(neg_idx), replace=False)))
            else:
                keep.extend(pos_idx)
                keep.extend(list(np.random.choice(neg_idx, len(pos_idx), replace=False)))

    if len(model2_zero_idx) > 0:
        pos_score_to_idx, neg_score_to_idx = {i: [] for i in range(n_grids)}, {i: [] for i in range(n_grids)}
        for i in model2_zero_idx:
            gap = model1_score_gap[i]
            if gap > 0:
                pos_score_to_idx[int(gap * n_grids)].append(i)
            else:
                neg_score_to_idx[int(-gap * n_grids)].append(i)
        for i in trange(n_grids):
            pos_idx = pos_score_to_idx[i]
            neg_idx = neg_score_to_idx[i]
            if len(pos_idx) > len(neg_idx):
                keep.extend(neg_idx)
                keep.extend(list(np.random.choice(pos_idx, len(neg_idx), replace=False)))
            else:
                keep.extend(pos_idx)
                keep.extend(list(np.random.choice(neg_idx, len(pos_idx), replace=False)))

    return keep