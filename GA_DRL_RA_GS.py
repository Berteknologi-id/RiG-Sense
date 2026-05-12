import numpy as np
import pandas as pd
import gym
from gym import spaces
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from collections import deque
import random
import ast
from collections import Counter
import time
from itertools import combinations
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass
from itertools import product

# Set seed
SEED = 42 #42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
network="bwsn"
filecat="bwsn_link"
# type="optimasi_normal_v2.1"
# directory=f"re_generate{network}_gsraga"
directory=f"re_generate{network}_gsraga_training"

# =========================
# 0b) SWITCH: grid or not
# =========================
GRID_SEARCH = False  # True: grid search -> final train with best HP; False: train with default HP

if not os.path.exists(directory):
    os.makedirs(directory)

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s.astype(np.float32), a.astype(np.int32), r.astype(np.float32), s2.astype(np.float32), d.astype(np.float32)

        # np.random.seed(SEED)  # Tambahkan seed di sini
        # np.random.default_rng(SEED)
        # indices = np.random.choice(len(self.buf), batch_size, replace=False)
        # batch = [self.buffer[i] for i in indices]

        # s, a, r, s2, d = map(np.array, zip(*batch))
        # return s.astype(np.float32), a.astype(np.int32), r.astype(np.float32), s2.astype(np.float32), d.astype(np.float32)

    def __len__(self):
        return len(self.buf)


def readDataFrameMultiRes(readLink,readLinkQuality,label,nodeSource,valDefault=0):
    dataResultLink={'Source':[]}
    dataResultLinkQuality={'Source':[]}

    dataCopyLink=readLink.loc[[label-1]].copy()
    dataCopyLinkQuality=readLinkQuality.loc[[label-1]].copy()
    
    for index,row in dataCopyLink.iterrows():
        for kr,ir in enumerate(nodeSource):
            dataResultLink['Source'].append(ir)
            dataResultLinkQuality['Source'].append(ir)
            node=ir
            for keys in dataCopyLink.keys():
                if keys!='NodeID' :
                    if keys not in dataResultLink:
                        dataResultLink[keys]=[]
                        dataResultLinkQuality[keys]=[]
                    val=valDefault
                    valQ=valDefault
                    if pd.isna(row[keys])!=True:
                        res=ast.literal_eval(row[keys])
                        restQ=ast.literal_eval(dataCopyLinkQuality[keys].values[0])
                        for irow,vrow in enumerate(res):
                            if node == vrow:
                                val=1
                                valQ=restQ[irow]
                    dataResultLink[keys].append(val)
                    dataResultLinkQuality[keys].append(valQ)

    resultLink=pd.DataFrame(dataResultLink)
    resultLinkQuality=pd.DataFrame(dataResultLinkQuality)
    return resultLink,resultLinkQuality
    # return resultLinkQuality


def timeEstimated(steps,readLink,sensor_location,label):
    readLink=readLink.loc[readLink['Source'].isin(sensor_location)]
    tr={}
    for index,item in readLink.iterrows():
        stepPos=item.loc[steps]
        if (stepPos == 1).any():
            first_position_index = stepPos[stepPos == 1].index[0]
            first_position_index_numeric = stepPos.index.get_loc(first_position_index)
        else:
            first_position_index_numeric=None
        tr[item['Source']]=first_position_index_numeric
    reposition=[tr[k] for k in sensor_location]
    return reposition
def timeEstimatedQ(steps,readLinkQuality,sensor_location,label):
    readLinkQuality=readLinkQuality.loc[readLinkQuality['Source'].isin(sensor_location)]
    tr={}
    for index,item in readLinkQuality.iterrows():
        stepPos=item.loc[steps]
        if (stepPos > 0).any():
            first_position_index = stepPos[stepPos > 0].index[0]
            # first_position_index_numeric = stepPos.index.get_loc(first_position_index)
            first_position_index_numeric = stepPos[first_position_index]
        else:
            first_position_index_numeric=None
        tr[item['Source']]=first_position_index_numeric
    reposition=[tr[k] for k in sensor_location]
    return reposition
# ---------------------------
# 1) Utilities
# ---------------------------
def validate_triplets(
    lokasi: Dict[int, List[int]],
    times_first: Dict[int, List[float]],
    quality_contaminant: Dict[int, List[float]],
    num_links: int = 58,
    strict: bool = True,
) -> None:
    errs = []
    for link in range(1, num_links + 1):
        a = lokasi.get(link, [])
        b = times_first.get(link, [])
        c = quality_contaminant.get(link, [])
        if not (len(a) == len(b) == len(c)):
            errs.append((link, len(a), len(b), len(c)))
    if errs:
        msg = "Inconsistency found (link, len(lokasi), len(times_first), len(quality)):\n" + \
              "\n".join(map(str, errs[:30]))
        if strict:
            raise ValueError(msg)
        else:
            print("[WARN]", msg)

def zscore_normalize(feats: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True)
    feats_n = (feats - mu) / (sd + eps)
    return feats_n.astype(np.float32), mu.squeeze(), sd.squeeze()

def compute_warmup_steps(memory_size: int, batch_size: int, warmup_ratio: float = 0.16) -> int:
    warmup = int(memory_size * warmup_ratio)
    warmup = max(warmup, 5 * batch_size)
    warmup = min(warmup, memory_size)
    return warmup

# ---------------------------
# 2) Risk weights per node (Version A)
# ---------------------------
def build_node_risk_weights(
    lokasi: Dict[int, List[int]],
    times_first: Dict[int, List[float]],
    quality_contaminant: Dict[int, List[float]],
    num_links: int = 58,
    w_exposure: float = 0.5,
    w_early: float = 0.3,
    w_quality: float = 0.2,
    eps: float = 1e-9
) -> Dict[int, float]:
    """
    Risk weight per NodeID from your dicts:
    - exposure: how often node appears across links
    - early: average 1/(t+1)
    - quality: average first-hit quality
    Output weights normalized to sum=1.
    """
    exposure = {}
    early_acc = {}
    early_cnt = {}
    qual_acc = {}
    qual_cnt = {}

    for link in range(1, num_links + 1):
        nodes = lokasi.get(link, [])
        ts = times_first.get(link, [])
        qs = quality_contaminant.get(link, [])
        m = min(len(nodes), len(ts), len(qs))
        nodes, ts, qs = nodes[:m], ts[:m], qs[:m]

        for node, t, q in zip(nodes, ts, qs):
            exposure[node] = exposure.get(node, 0) + 1
            early_acc[node] = early_acc.get(node, 0.0) + (1.0 / (float(t) + 1.0))
            early_cnt[node] = early_cnt.get(node, 0) + 1
            qual_acc[node] = qual_acc.get(node, 0.0) + float(q)
            qual_cnt[node] = qual_cnt.get(node, 0) + 1

    nodes_all = sorted(exposure.keys())
    exp_arr = np.array([exposure[n] for n in nodes_all], dtype=float)
    early_arr = np.array([early_acc[n] / max(early_cnt[n], 1) for n in nodes_all], dtype=float)
    qual_arr = np.array([qual_acc[n] / max(qual_cnt[n], 1) for n in nodes_all], dtype=float)

    def minmax(x: np.ndarray) -> np.ndarray:
        mn, mx = float(x.min()), float(x.max())
        if abs(mx - mn) < eps:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    exp_n = minmax(exp_arr)
    early_n = minmax(early_arr)
    qual_n = minmax(qual_arr)

    score = w_exposure * exp_n + w_early * early_n + w_quality * qual_n
    score = np.maximum(score, 0.0)
    score = score / (score.sum() + eps)

    return {node: float(w) for node, w in zip(nodes_all, score)}

# ---------------------------
# 3) Graph-aware link features (FDG upgrade) from dicts
# ---------------------------
def build_link_features_fdg(
    lokasi: Dict[int, List[int]],
    times_first: Dict[int, List[float]],          # minutes
    quality_contaminant: Dict[int, List[float]],
    num_links: int = 58,
    node_weights: Optional[Dict[int, float]] = None,
    t_max_minutes: Optional[float] = None,
    eps: float = 1e-9
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Returns:
    - df features per link
    - feats array shape (num_links, F)
    """
    # total unique nodes
    all_nodes = set()
    for nodes in lokasi.values():
        all_nodes.update(nodes)
    total_nodes = max(len(all_nodes), 1)

    # t_max from data if not provided
    if t_max_minutes is None:
        mx = 0.0
        for ts in times_first.values():
            for t in ts:
                mx = max(mx, float(t))
        t_max_minutes = max(mx, 1.0)

    rows = []
    for link in range(1, num_links + 1):
        nodes = lokasi.get(link, [])
        ts = times_first.get(link, [])
        qs = quality_contaminant.get(link, [])

        m = min(len(nodes), len(ts), len(qs))
        nodes = nodes[:m]
        ts = [float(x) for x in ts[:m]]
        qs = [float(x) for x in qs[:m]]

        cov_count = len(nodes)
        cov_ratio = cov_count / total_nodes

        if node_weights is not None and cov_count > 0:
            risk_cov_sum = sum(node_weights.get(n, 0.0) for n in nodes)
        else:
            risk_cov_sum = 0.0

        if cov_count == 0:
            t_min = t_max_minutes
            t_mean = t_max_minutes
            early_score_mean = 0.0
            t_min_norm = 1.0
            t_mean_norm = 1.0
            q_mean = 0.0
            q_max = 0.0
            quality_early_mean = 0.0
        else:
            t_min = float(min(ts))
            t_mean = float(sum(ts) / cov_count)
            early_score_mean = float(sum(1.0 / (t + 1.0) for t in ts) / cov_count)  # faster => bigger
            t_min_norm = float(t_min / (t_max_minutes + eps))
            t_mean_norm = float(t_mean / (t_max_minutes + eps))
            q_mean = float(sum(qs) / cov_count)
            q_max = float(max(qs))
            quality_early_mean = float(sum((q / (t + 1.0)) for q, t in zip(qs, ts)) / cov_count)

        exposure_prior = float(cov_count * early_score_mean)

        rows.append({
            "link": link,
            "cov_count": cov_count,
            "cov_ratio": cov_ratio,
            "risk_cov_sum": risk_cov_sum,
            "t_min_minute": t_min,
            "t_mean_minute": t_mean,
            "t_min_norm": t_min_norm,
            "t_mean_norm": t_mean_norm,
            "early_score_mean": early_score_mean,
            "q_mean": q_mean,
            "q_max": q_max,
            "quality_early_mean": quality_early_mean,
            "exposure_prior": exposure_prior,
        })

    df = pd.DataFrame(rows)

    feature_cols = [
        "cov_count", "cov_ratio", "risk_cov_sum",
        "t_min_norm", "t_mean_norm", "early_score_mean",
        "q_mean", "q_max", "quality_early_mean",
        "exposure_prior",
    ]
    feats = df[feature_cols].to_numpy(dtype=np.float32)
    return df, feats, feature_cols


# ---------------------------
# 4) Risk-aware objective (Version A) from dicts
# ---------------------------
@dataclass
class DetectionStats:
    t_min: Optional[float]   # minutes
    q_at_tmin: Optional[float]

def build_node_to_linkinfo(
    lokasi: Dict[int, List[int]],
    times_first: Dict[int, List[float]],
    quality_contaminant: Dict[int, List[float]],
    num_links: int = 58
) -> Dict[int, List[Tuple[int, float, float]]]:
    """
    Build index:
      node -> list of (link, time_minute, quality_firsthit)
    from your aligned lists.
    """
    node_index: Dict[int, List[Tuple[int, float, float]]] = {}
    for link in range(1, num_links + 1):
        nodes = lokasi.get(link, [])
        ts = times_first.get(link, [])
        qs = quality_contaminant.get(link, [])
        m = min(len(nodes), len(ts), len(qs))
        for node, t, q in zip(nodes[:m], ts[:m], qs[:m]):
            node_index.setdefault(int(node), []).append((link, float(t), float(q)))
    return node_index

def compute_detection_for_node(
    node: int,
    selected_links: Set[int],
    node_index: Dict[int, List[Tuple[int, float, float]]]
) -> DetectionStats:
    """
    For a node (injection source), detection time is min time among selected links that cover this node.
    q_at_tmin = quality of the link achieving that min time (tie -> max quality among min-time ties).
    """
    candidates = node_index.get(node, [])
    best_t = None
    best_q = None
    for link, t, q in candidates:
        if link in selected_links:
            if best_t is None or t < best_t:
                best_t = t
                best_q = q
            elif t == best_t and best_q is not None and q > best_q:
                best_q = q
    return DetectionStats(t_min=best_t, q_at_tmin=best_q)

def risk_aware_objective_versionA(
    selected_links: Set[int],
    nodes_all: List[int],
    node_weights: Dict[int, float],
    node_index: Dict[int, List[Tuple[int, float, float]]],
    t_max_minutes: float,
    q_max_global: float,
    lambda_time: float = 0.6,
    beta_quality: float = 0.2,
    miss_penalty: float = 1.0,
) -> float:
    """
    Objective J(S) = sum_node w(node) * R_node(S)

    R_node(S):
      if detected:
         base = 1 - lambda_time * (t / t_max)
         plus = beta_quality * (q / q_max)
         R = base + plus
      else:
         R = - miss_penalty
    """
    total = 0.0
    for node in nodes_all:
        w = node_weights.get(node, 0.0)
        det = compute_detection_for_node(node, selected_links, node_index)
        if det.t_min is None:
            total += w * (-miss_penalty)
        else:
            t_norm = det.t_min / max(t_max_minutes, 1e-9)
            q_norm = (det.q_at_tmin / max(q_max_global, 1e-9)) if det.q_at_tmin is not None else 0.0
            r = (1.0 - lambda_time * t_norm) + (beta_quality * q_norm)
            total += w * r
    return float(total)

def evaluate_selected_links_metrics(
    selected_links: Set[int],
    nodes_all: List[int],
    node_index: Dict[int, List[Tuple[int, float, float]]],
) -> dict:
    """
    Output:
    - coverage_count: jumlah node terdeteksi (unik)
    - coverage_ratio: persen coverage dari total nodes_all
    - detected_nodes: jumlah node terdeteksi
    - undetected_nodes: jumlah node tidak terdeteksi
    - time_total_detected: total menit deteksi (hanya node terdeteksi)
    - time_avg_detected: rata-rata menit deteksi (hanya node terdeteksi)
    - time_min_detected / time_max_detected: ringkasan waktu (opsional)
    """
    detected_times = []
    detected_set = set()

    for node in nodes_all:
        best_t = None
        for link, t, q in node_index.get(node, []):
            if link in selected_links:
                if best_t is None or t < best_t:
                    best_t = t

        if best_t is not None:
            detected_set.add(node)
            detected_times.append(float(best_t))

    total_nodes = max(len(nodes_all), 1)
    coverage_count = len(detected_set)
    coverage_ratio = (coverage_count / total_nodes) * 100.0

    detected_nodes = coverage_count
    undetected_nodes = total_nodes - detected_nodes

    if detected_nodes > 0:
        time_total = float(np.sum(detected_times))
        time_avg = float(np.mean(detected_times))
        time_min = float(np.min(detected_times))
        time_max = float(np.max(detected_times))
    else:
        time_total = 0.0
        time_avg = 0.0
        time_min = 0.0
        time_max = 0.0

    return {
        "coverage_count": coverage_count,
        "coverage_ratio_percent": coverage_ratio,
        "detected_nodes": detected_nodes,
        "undetected_nodes": undetected_nodes,
        "time_total_detected_min": time_total,
        "time_avg_detected_min": time_avg,
        "time_min_detected_min": time_min,
        "time_max_detected_min": time_max,
    }


# ---------------------------
# 5) Environment (choose K links out of 58)
# ---------------------------
class LinkPlacementEnvV_A:
    def __init__(
        self,
        num_links: int,
        K: int,
        link_feats: np.ndarray,          # shape (E,F) where E=58 in order link 1..58
        nodes_all: List[int],
        node_weights: Dict[int, float],
        node_index: Dict[int, List[Tuple[int, float, float]]],
        t_max_minutes: float,
        q_max_global: float,
        lambda_time: float = 0.6,
        beta_quality: float = 0.2,
        miss_penalty: float = 1.0,
        invalid_penalty: float = -0.05,
    ):
        self.E = num_links
        self.K = K
        self.link_feats = link_feats.astype(np.float32)
        self.nodes_all = nodes_all
        self.node_weights = node_weights
        self.node_index = node_index
        self.t_max_minutes = t_max_minutes
        self.q_max_global = q_max_global
        self.lambda_time = lambda_time
        self.beta_quality = beta_quality
        self.miss_penalty = miss_penalty
        self.invalid_penalty = invalid_penalty

        self.reset()

    def reset(self) -> np.ndarray:
        self.placement = np.zeros(self.E, dtype=np.int32)  # internal index 0..57
        self.selected_links: Set[int] = set()              # real link id 1..58
        self.k_selected = 0
        self.last_obj = 0.0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        # state = [placement(58), link_feats(58*F), progress(1)]
        progress = np.array([self.k_selected / max(self.K, 1)], dtype=np.float32)
        return np.concatenate([
            self.placement.astype(np.float32),
            self.link_feats.reshape(-1),
            progress
        ], axis=0)

    def valid_actions_mask(self) -> np.ndarray:
        # 1 = valid, 0 = invalid (already chosen)
        return (self.placement == 0).astype(np.float32)

    def objective(self) -> float:
        return risk_aware_objective_versionA(
            selected_links=self.selected_links,
            nodes_all=self.nodes_all,
            node_weights=self.node_weights,
            node_index=self.node_index,
            t_max_minutes=self.t_max_minutes,
            q_max_global=self.q_max_global,
            lambda_time=self.lambda_time,
            beta_quality=self.beta_quality,
            miss_penalty=self.miss_penalty,
        )

    def step(self, action_internal: int) -> Tuple[np.ndarray, float, bool, dict]:
        # action_internal in [0..57] -> link_id = action_internal + 1
        if action_internal < 0 or action_internal >= self.E:
            return self._get_state(), self.invalid_penalty, False, {"invalid": True}

        if self.placement[action_internal] == 1:
            return self._get_state(), self.invalid_penalty, False, {"invalid": True}

        link_id = action_internal + 1

        # apply
        self.placement[action_internal] = 1
        self.selected_links.add(link_id)
        self.k_selected += 1

        # incremental reward
        new_obj = self.objective()
        reward = new_obj - self.last_obj
        self.last_obj = new_obj

        done = (self.k_selected >= self.K)
        info = {"objective": new_obj, "k_selected": self.k_selected, "selected_links": sorted(self.selected_links)}
        return self._get_state(), float(reward), done, info


# ---------------------------
# 6) DQN (Keras)
# ---------------------------
def build_q_network(input_dim: int, num_actions: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), dtype=tf.float32)
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(num_actions, activation=None)(x)
    return keras.Model(inputs, outputs)

def masked_argmax(q_values: np.ndarray, mask: np.ndarray) -> int:
    masked = q_values - (1.0 - mask) * 1e9
    return int(masked.argmax())

def train_dqn(
    env: LinkPlacementEnvV_A,
    episodes: int,
    gamma: float,
    lr: float,
    batch_size: int,
    replay_capacity: int,
    warmup_steps: int,
    train_every: int,
    target_update_every: int,
    eps_start: float,
    eps_end: float,
    epsilon_decay: float,
    gradient_clip: float,
    seed: int,
    verbose_every: int = 50,
    log_csv_path: Optional[str] = "episode_log.csv",
):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    num_actions = env.E
    state_dim = env.reset().shape[0]

    online = build_q_network(state_dim, num_actions)
    target = build_q_network(state_dim, num_actions)
    target.set_weights(online.get_weights())

    opt = keras.optimizers.Adam(learning_rate=lr)
    rb = ReplayBuffer(capacity=replay_capacity)

    best_obj = -1e18
    best_links = []
    best_ep = None
    global_step = 0
    eps = eps_start
    best_metrics = {}
    episode_logs = [] 

    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False

        while not done:
            global_step += 1

            mask = env.valid_actions_mask()
            if random.random() < eps:
                valid_actions = np.where(mask > 0.5)[0]
                a = int(np.random.choice(valid_actions))
            else:
                q = online(s[None, :], training=False).numpy()[0]
                a = masked_argmax(q, mask)

            s2, r, done, info = env.step(a)
            rb.add(s, a, r, s2, done)
            s = s2

            # epsilon decay (multiplicative)
            eps = max(eps_end, eps * epsilon_decay)

            # training
            if len(rb) >= warmup_steps and (global_step % train_every == 0):
                sb, ab, rb_, s2b, db = rb.sample(batch_size)

                # next-state valid action masks derived from placement slice
                placement_next = s2b[:, :env.E]
                mask_next = (placement_next == 0).astype(np.float32)

                q_next = target(s2b, training=False).numpy()
                q_next_masked = q_next - (1.0 - mask_next) * 1e9
                max_next = q_next_masked.max(axis=1)

                y = rb_ + (1.0 - db) * gamma * max_next

                with tf.GradientTape() as tape:
                    q_all = online(sb, training=True)
                    q_a = tf.gather(q_all, ab[:, None], batch_dims=1)[:, 0]
                    loss = tf.reduce_mean(tf.square(y - q_a))

                grads = tape.gradient(loss, online.trainable_variables)

                # gradient clipping
                if gradient_clip is not None and gradient_clip > 0:
                    grads, _ = tf.clip_by_global_norm(grads, gradient_clip)

                opt.apply_gradients(zip(grads, online.trainable_variables))

            if global_step % target_update_every == 0:
                target.set_weights(online.get_weights())

        final_obj = float(info.get("objective", env.last_obj))
        selected = info.get("selected_links", [])
        selected_set = set(selected)
        
        # hitung metrics episode ini
        ep_metrics = evaluate_selected_links_metrics(
            selected_links=selected_set,
            nodes_all=env.nodes_all,
            node_index=env.node_index,
        )
        # simpan log episode
        episode_logs.append({
            "row_type": "EPISODE",
            "episode": ep,
            "objective": final_obj,
            "epsilon_end": float(eps),
            "selected_links": " ".join(map(str, selected)),
            "k_selected": len(selected),

            "coverage_count": ep_metrics["coverage_count"],
            "coverage_ratio_percent": ep_metrics["coverage_ratio_percent"],
            "detected_nodes": ep_metrics["detected_nodes"],
            "undetected_nodes": ep_metrics["undetected_nodes"],

            "time_total_detected_min": ep_metrics["time_total_detected_min"],
            "time_avg_detected_min": ep_metrics["time_avg_detected_min"],
            "time_min_detected_min": ep_metrics["time_min_detected_min"],
            "time_max_detected_min": ep_metrics["time_max_detected_min"],
        })
        if final_obj > best_obj: 
            best_obj = final_obj
            best_links = selected
            best_metrics = ep_metrics
            best_ep=ep

        if verbose_every and (ep % verbose_every == 0):
            print(f"[ep {ep:4d}] obj={final_obj:.6f} best={best_obj:.6f} eps={eps:.3f} selected(best)={best_links}")

    if log_csv_path:
        df_log = pd.DataFrame(episode_logs)
        # Pastikan kolom konsisten (kalau episode_logs kosong)
        if df_log.empty:
            df_log = pd.DataFrame(columns=[
                "row_type",
                "episode","objective","epsilon_end","selected_links","k_selected",
                "coverage_count","coverage_ratio_percent","detected_nodes","undetected_nodes",
                "time_total_detected_min","time_avg_detected_min","time_min_detected_min","time_max_detected_min"
            ])
        
        # Row BEST (ringkasan terbaik)
        best_row = {
            "row_type": "BEST",
            "episode": best_ep,
            "objective": best_obj,
            "epsilon_end": "",
            "selected_links": " ".join(map(str, best_links)),
            "k_selected": len(best_links),

            "coverage_count": best_metrics.get("coverage_count", ""),
            "coverage_ratio_percent": best_metrics.get("coverage_ratio_percent", ""),
            "detected_nodes": best_metrics.get("detected_nodes", ""),
            "undetected_nodes": best_metrics.get("undetected_nodes", ""),

            "time_total_detected_min": best_metrics.get("time_total_detected_min", ""),
            "time_avg_detected_min": best_metrics.get("time_avg_detected_min", ""),
            "time_min_detected_min": best_metrics.get("time_min_detected_min", ""),
            "time_max_detected_min": best_metrics.get("time_max_detected_min", ""),
        }

        # Row kosong sebagai pemisah
        blank_row = {col: "" for col in df_log.columns}
        blank_row["row_type"] = "BLANK"

        # Row marker supaya jelas mulai log episode
        marker_row = {col: "" for col in df_log.columns}
        marker_row["row_type"] = "EPISODE_LOG_START"

        # Gabungkan: BEST -> blank -> marker -> logs
        df_out = pd.concat(
            [
                pd.DataFrame([best_row]),
                pd.DataFrame([blank_row]),
                pd.DataFrame([marker_row]),
                df_log
            ],
            ignore_index=True
        )

        df_out.to_csv(log_csv_path, index=False)

    return {
        "best_objective": best_obj,
        "best_links": best_links,
        "best_metrics": best_metrics,
        # "episode_logs": episode_logs,
        "online_model": online
    }

# ---------------------------
# 7) GRID SEARCH (optional)
# ---------------------------
def grid_search_hyperparams(
    env_builder_fn: Callable[[], LinkPlacementEnvV_A],
    episodes_per_trial: int,
    seeds: List[int],
    learning_rates: List[float],
    gammas: List[float],
    batch_sizes: List[int],
    memory_sizes: List[int],
    epsilon_decays: List[float],
    gradient_clips: List[Optional[float]],
    warmup_ratio: float,
    eps_start: float,
    eps_end: float,
    target_update_every: int,
    train_every: int,
    progress_every: int = 10,
) -> pd.DataFrame:
    combos = list(product(
        learning_rates, gammas, batch_sizes, memory_sizes, epsilon_decays, gradient_clips
    ))
    print("Total grid combinations:", len(combos))

    trials = []
    best_seen = -1e18
    t0 = time.time()

    for idx, (lr, gm, bs, mem, epsdec, gclip) in enumerate(combos, start=1):
        warmup_steps = compute_warmup_steps(mem, bs, warmup_ratio)

        scores = []
        for sd in seeds:
            env = env_builder_fn()
            res = train_dqn(
                env=env,
                episodes=episodes_per_trial,
                gamma=gm,
                lr=lr,
                batch_size=bs,
                replay_capacity=mem,
                warmup_steps=warmup_steps,
                train_every=train_every,
                target_update_every=target_update_every,
                eps_start=eps_start,
                eps_end=eps_end,
                epsilon_decay=epsdec,
                gradient_clip=gclip,
                seed=sd,
                verbose_every=0,
                log_csv_path=None,  # grid: jangan export episode
            )
            scores.append(float(res["best_objective"]))

        score_mean = float(np.mean(scores))
        score_std = float(np.std(scores))

        trials.append({
            "trial": idx,
            "lr": lr,
            "gamma": gm,
            "batch_size": bs,
            "memory_size": mem,
            "epsilon_decay": epsdec,
            "gradient_clip": gclip,
            "episodes_per_trial": episodes_per_trial,
            "warmup_steps": warmup_steps,
            "seeds": ",".join(map(str, seeds)),
            "score_mean": score_mean,
            "score_std": score_std,
        })

        if score_mean > best_seen:
            best_seen = score_mean

        if progress_every and (idx % progress_every == 0):
            elapsed = time.time() - t0
            print(f"[{idx}/{len(combos)}] elapsed={elapsed:.1f}s best_mean_so_far={best_seen:.6f}")

    df = pd.DataFrame(trials).sort_values(["score_mean", "score_std"], ascending=[False, True]).reset_index(drop=True)
    return df



if __name__ == "__main__":
    start_time = time.time()
    # =========================
    # 0c) Core configs
    # =========================

    K_SENSORS=10
    
    #bwsn
    LEARNING_RATE = 0.0005
    GAMMA = 0.95
    BATCH_SIZE = 32
    MEMORY_SIZE = 2000
    EPSILON_DECAY = 0.999      
    GRADIENT_CLIP = None
    #zj
    # LEARNING_RATE = 0.001
    # GAMMA = 0.95
    # BATCH_SIZE = 32
    # MEMORY_SIZE = 2000
    # EPSILON_DECAY = 0.999      
    # GRADIENT_CLIP = 5.0
    #jilin
    # LEARNING_RATE = 0.001
    # GAMMA = 0.98
    # BATCH_SIZE = 32
    # MEMORY_SIZE = 2000
    # EPSILON_DECAY = 0.995        
    # GRADIENT_CLIP = None
    #fos
    # LEARNING_RATE = 0.0005
    # GAMMA = 0.95
    # BATCH_SIZE = 32
    # MEMORY_SIZE = 2000
    # EPSILON_DECAY = 0.995        
    # GRADIENT_CLIP = 1.0
    #default
    # LEARNING_RATE = 0.0005
    # GAMMA = 0.95
    # BATCH_SIZE = 32
    # MEMORY_SIZE = 2000
    # EPSILON_DECAY = 0.99        
    # GRADIENT_CLIP = 5.0

    EPS_START = 1.0
    EPS_END = 0.05   
    EPISODES = 19000
    TARGET_UPDATE_EVERY = 200
    TRAIN_EVERY = 1
    
    # warmup heuristic: warmup = max(5*batch_size, warmup_ratio*memory_size), capped at memory_size
    WARMUP_RATIO = 0.16

    # Risk-aware objective knobs (feel free to tune)
    LAMBDA_TIME = 0.6
    BETA_QUALITY = 0.2
    MISS_PENALTY = 1.0
    INVALID_PENALTY = -0.05

    OUTPUTVERS="_19k"
    # File outputs
    FEATURES_CSV = f"{directory}/link_features_fdg{OUTPUTVERS}.csv"
    GRID_RESULTS_CSV = f"{directory}/grid_search_results{OUTPUTVERS}.csv"
    FINAL_EPISODE_LOG_CSV = f"{directory}/episode_results{OUTPUTVERS}.csv"
    MODEL_OUT = f"{directory}/rasp_dqn_model_best{OUTPUTVERS}.keras"

    # =========================
    # 0d) Grid search space
    # =========================
    learning_rates = [0.0005, 0.001, 0.005]
    gammas = [0.95, 0.98, 0.99]
    batch_sizes = [32, 64]
    memory_sizes = [2000, 5000, 10000]
    epsilon_decays = [0.99, 0.995, 0.999]
    gradient_clips = [None, 1.0, 5.0]

    GRID_EPISODES_PER_TRIAL = 100
    GRID_SEEDS = [SEED]  # 1-2 seeds (lebih banyak = lebih stabil, lebih berat)
    GRID_PROGRESS_EVERY = 10

    logfilename= f"{directory}/DRL_final_{network}.txt"
    with open(logfilename, 'w') as logfile:
        logfile.writelines("load file CSV\n")
        directoryA= os.path.join("output_simulation/time_contamination", f"{filecat}.csv")
        directoryB= os.path.join("output_simulation/time_contamination", f"{filecat}_quality.csv")
        readLink = pd.read_csv(directoryA, delimiter=';', on_bad_lines='skip')
        readLinkQuality = pd.read_csv(directoryB, delimiter=';', on_bad_lines='skip')


        logfile.writelines("Read file CSV\n")
        steps = [col for col in readLink.columns if col.startswith('links_step')]
        step_data = {col: np.array([x if pd.isna(x)!=True else 0 for x in readLink[col]]) for col in steps}
        all_values = []
        for key, values in step_data.items():
            for val in values:
                all_values.extend(val[1:-1].split(","))
        all_values = [int(x.strip()) for x in all_values if x!='']
        unique_values = list(set(all_values))
        nodeSources=[node for node in readLink['NodeID']]
        K_SENSORS=int(np.ceil(len(unique_values)/100*10))
        # K_SENSORS=round(len(unique_values)*10/100)

        sensor_coverage ={i:[] for i in unique_values}
        sensor_times ={i:[] for i in unique_values}
        sensor_quality ={i:[] for i in unique_values}
        tempQuality=pd.DataFrame([])
        # row=1
        for source in nodeSources:
            resultLinkO,readLinkQualityO=readDataFrameMultiRes(readLink,readLinkQuality,source,unique_values,valDefault=0)
            if tempQuality.empty==True:
                tempQuality=readLinkQualityO
            else:
                for col in readLink.columns:
                    if col.startswith('links_step'):
                        tempQuality[col]=tempQuality[col]+readLinkQualityO[col]
            for link in unique_values:
                estimated=timeEstimated(steps,resultLinkO,[link],source)[0]
                estimatedQ=timeEstimatedQ(steps,readLinkQualityO,[link],source)[0]
                if estimated!=None:
                    sensor_coverage[link].append(source)
                    sensor_times[link].append(estimated)
                    sensor_quality[link].append(estimatedQ)

        # tempQuality=tempQuality.drop(columns=['Source']).values
        
        logfile.writelines("Config DRL CSV\n")
        
        NUM_LINKS=len(unique_values)
        logfile.writelines("Preparing Graph Awarness\n")
        
        # 1) validate input dicts
        validate_triplets(sensor_coverage, sensor_times, sensor_quality, num_links=NUM_LINKS, strict=True)
        
        # 2) build node risk weights 
        node_w = build_node_risk_weights(sensor_coverage, sensor_times, sensor_quality, num_links=NUM_LINKS)
        nodes_all = sorted(node_w.keys())
        
        # 3) build node->(link,t,q) index
        node_index = build_node_to_linkinfo(sensor_coverage, sensor_times, sensor_quality, num_links=NUM_LINKS)

        # derive t_max_minutes and q_max_global from dicts for normalization in objective
        t_max_minutes = 0.0
        q_max_global = 0.0
        for link in range(1, NUM_LINKS + 1):
            for t in sensor_times.get(link, []):
                t_max_minutes = max(t_max_minutes, float(t))
            for q in sensor_quality.get(link, []):
                q_max_global = max(q_max_global, float(q))
        t_max_minutes = max(t_max_minutes, 1.0)
        q_max_global = max(q_max_global, 1.0)

        # 4) graph-aware link feats (FDG upgrade) + normalize
        df_feat, feats, feat_cols = build_link_features_fdg(
            sensor_coverage, sensor_times, sensor_quality,
            num_links=NUM_LINKS,
            node_weights=node_w,
            t_max_minutes=t_max_minutes
        )
        feats_norm, mu, sd = zscore_normalize(feats)

        # optional: export features table
        df_feat.to_csv(FEATURES_CSV, index=False)
        # np.save("link_features_fdg.npy", feats_norm)

        # env builder
        def env_builder_fn() -> LinkPlacementEnvV_A:
            return LinkPlacementEnvV_A(
                num_links=NUM_LINKS,
                K=K_SENSORS,
                link_feats=feats_norm,
                nodes_all=nodes_all,
                node_weights=node_w,
                node_index=node_index,
                t_max_minutes=t_max_minutes,
                q_max_global=q_max_global,
                lambda_time=LAMBDA_TIME,
                beta_quality=BETA_QUALITY,
                miss_penalty=MISS_PENALTY,
                invalid_penalty=INVALID_PENALTY,
            )
        if GRID_SEARCH:
            print("\n=== GRID SEARCH: ON ===")
            gs_df = grid_search_hyperparams(
                env_builder_fn=env_builder_fn,
                episodes_per_trial=GRID_EPISODES_PER_TRIAL,
                seeds=GRID_SEEDS,
                learning_rates=learning_rates,
                gammas=gammas,
                batch_sizes=batch_sizes,
                memory_sizes=memory_sizes,
                epsilon_decays=epsilon_decays,
                gradient_clips=gradient_clips,
                warmup_ratio=WARMUP_RATIO,
                eps_start=EPS_START,
                eps_end=EPS_END,
                target_update_every=TARGET_UPDATE_EVERY,
                train_every=TRAIN_EVERY,
                progress_every=GRID_PROGRESS_EVERY
            )
            gs_df.to_csv(GRID_RESULTS_CSV, index=False)
            print("Saved grid results:", GRID_RESULTS_CSV)
            print(gs_df.head(10)[["lr","gamma","batch_size","memory_size","epsilon_decay","gradient_clip","score_mean","score_std"]])

            best = gs_df.iloc[0].to_dict()
            best_lr = float(best["lr"])
            best_gamma = float(best["gamma"])
            best_bs = int(best["batch_size"])
            best_mem = int(best["memory_size"])
            best_epsdec = float(best["epsilon_decay"])
            best_gclip = best["gradient_clip"]
            if pd.isna(best_gclip):
                best_gclip = None

            warmup_steps = compute_warmup_steps(best_mem, best_bs, WARMUP_RATIO)
        
            print("\n=== FINAL TRAIN (best HP) ===")
            print(best)
            logfile.writelines(f'{best}\n')
            # final_env = env_builder_fn()
            # result = train_dqn(
            #     env=final_env,
            #     episodes=EPISODES,
            #     gamma=best_gamma,
            #     lr=best_lr,
            #     batch_size=best_bs,
            #     replay_capacity=best_mem,
            #     warmup_steps=warmup_steps,
            #     train_every=TRAIN_EVERY,
            #     target_update_every=TARGET_UPDATE_EVERY,
            #     eps_start=EPS_START,
            #     eps_end=EPS_END,
            #     epsilon_decay=best_epsdec,
            #     gradient_clip=best_gclip,
            #     seed=SEED,
            #     verbose_every=50,
            #     log_csv_path=FINAL_EPISODE_LOG_CSV
            # )
            # result["online_model"].save(MODEL_OUT)
        else:
            warmup_steps = compute_warmup_steps(MEMORY_SIZE, BATCH_SIZE, WARMUP_RATIO)
            env = env_builder_fn()
            result = train_dqn(
                env=env,
                episodes=EPISODES,
                gamma=GAMMA,
                lr=LEARNING_RATE,
                batch_size=BATCH_SIZE,
                replay_capacity=MEMORY_SIZE,
                warmup_steps=warmup_steps,
                train_every=TRAIN_EVERY,
                target_update_every=TARGET_UPDATE_EVERY,
                eps_start=EPS_START,
                eps_end=EPS_END,
                epsilon_decay=EPSILON_DECAY,
                gradient_clip=GRADIENT_CLIP,
                seed=SEED,
                verbose_every=50,
                log_csv_path=FINAL_EPISODE_LOG_CSV
            )
            result["online_model"].save(MODEL_OUT)
        # save model
        fr_time = time.time() - start_time
        logfile.writelines(f'Time Execution : {fr_time:.2f} Seconds\n')
        print("\n=== RESULT SUMMARY ===")
        print("Best objective:", result["best_objective"])
        print("Best sensor links (1..58):", result["best_links"])
        print("Best metrics:", result.get("best_metrics", {}))
        print("Saved episode CSV:", FINAL_EPISODE_LOG_CSV)
        print("Saved model:", MODEL_OUT)
 