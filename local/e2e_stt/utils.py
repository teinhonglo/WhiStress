import re
import json
import os

import numpy as np
from tqdm import tqdm
import argparse

def get_stats(word_info, raw_list):
    start_time, end_time = word_info
    st_idx = int(start_time * 100)
    ed_idx = int(end_time * 100)

    stats_np = np.array(raw_list[st_idx: ed_idx])
    stats_np = stats_np[np.nonzero(stats_np)]

    number = len(stats_np)

    if number == 0:
        summ, mean, std, median, mad, maximum, minimum = 0, 0, 0, 0, 0, 0, 0
    else:
        summ = np.sum(stats_np)
        mean = np.mean(stats_np)
        std = np.std(stats_np)
        median = np.median(stats_np)
        mad = np.sum(np.absolute(stats_np - mean)) / number
        maximum = np.max(stats_np)
        minimum = np.min(stats_np)

    return [mean, std, median, mad, maximum, minimum]

def get_delivery_features(all_info):
    # f0
    f0_list = all_info["feats"]["f0_list"]
    # energy
    energy_list = all_info["feats"]["energy_rms_list"]
    
    delivery_feats = []

    #print("word-level length", len(all_info[uttid]["word_ctm"]))
    for i in range(len(all_info["word_ctm"])):
        word, start_time, duration, confidence = all_info["word_ctm"][i]
        start_time = float(start_time)
        duration = float(duration)
        confidence = float(confidence)
        end_time = start_time + duration

        word_info = [start_time, end_time]
        # f0
        f0_feats = get_stats(word_info, f0_list)
        # energy
        energy_feats = get_stats(word_info, energy_list)

        if i < len(all_info["word_ctm"]) - 1:
            following_word, following_start_time, following_duration, following_confidence = all_info["word_ctm"][i+1]
            following_start_time = float(following_start_time)
            following_duration = float(following_duration)
            following_pause_length = following_start_time - end_time
        else:
            following_pause_length = 0.0

        delivery_feats.append([duration] + f0_feats + energy_feats + [following_pause_length, confidence])

    return delivery_feats
