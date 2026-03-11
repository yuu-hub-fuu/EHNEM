"""
model_ehnem_v2.py

EHNEM 模型修复版 — 借鉴 DualHGCN 的稳定超图卷积实现

核心改动 (对比 v1):
  1. Pre-compute G matrix: 仿 DualHGCN 的 generate_G_from_H，
     将 incidence matrix H 预计算为传播矩阵 G = D^{-1/2} H W B^{-1} H^T D^{-1/2}
     卷积变成简单的 X' = σ(G · X · Θ)，数值稳定

  2. Residual connection: HGNN 每层输出 = conv(X) + X，
     即使超图质量差，原始 ESE 特征也不会被冲掉

  3. Incidence matrix 构建:
     - 每个 event node 自带一条 self-hyperedge（保证孤立节点也有信息流）
     - 支持 edge_dropout（训练时随机丢弃超边，缩小 GT/predicted 分布差距）

  4. 训练时用 GT causal pairs（论文 §3.2.1 明确说的），
     但加 edge_dropout 模拟推理时的噪声超图

  5. 分类器加 gating 机制: 学习 ESE vs HGNN 特征的权重，
     避免垃圾 HGNN 输出淹没好的 ESE 特征
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel


# ── Focal Loss (保留，可选用) ─────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, num_classes=2):
        super().__init__()
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        else:
            self.alpha = torch.tensor(
                [1 - alpha] + [alpha] * (num_classes - 1), dtype=torch.float)
        self.gamma = gamma

    def forward(self, logits, labels):
        alpha = self.alpha.to(logits.device)
        logits = logits.clamp(-30, 30)
        log_p = F.log_softmax(logits, dim=-1)
        p     = log_p.exp()
        p_t   = p.gather(1, labels.view(-1, 1)).squeeze(1).clamp(min=1e-8)
        lp_t  = log_p.gather(1, labels.view(-1, 1)).squeeze(1).clamp(min=-100)
        a_t   = alpha.gather(0, labels)
        loss  = -a_t * (1 - p_t) ** self.gamma * lp_t
        return loss.mean()


class WeightedCELoss(nn.Module):
    def __init__(self, pos_weight=0.75):
        super().__init__()
        self.register_buffer('weight', torch.tensor([1.0 - pos_weight, pos_weight]))

    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels, weight=self.weight)


def _build_loss_fn(args):
    loss_type = getattr(args, 'loss_type', 'ce')
    cw = getattr(args, 'class_weight', 0.75)
    if loss_type == 'focal':
        return FocalLoss(alpha=cw, gamma=args.gamma, num_classes=2)
    else:
        return WeightedCELoss(pos_weight=cw)


# ── Stage 1: ESE (不变) ───────────────────────────────────────
class ESEModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(args.model_name_or_path)
        d = self.roberta.config.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Dropout(args.dropout_ese),
            nn.Linear(d, 2),
        )
        self.loss_fn = _build_loss_fn(args)

    def _pool_span(self, hidden, spans):
        out = []
        for i, (s, e) in enumerate(spans):
            e = max(e, s + 1)
            out.append(hidden[i, s:e, :].mean(dim=0))
        return torch.stack(out)

    def forward(self, enc_input_ids, enc_mask_ids,
                e1_spans, e2_spans, labels=None):
        hidden = self.roberta(
            enc_input_ids, attention_mask=enc_mask_ids,
        ).last_hidden_state

        e1_emb = self._pool_span(hidden, e1_spans)
        e2_emb = self._pool_span(hidden, e2_spans)
        logits = self.mlp(torch.cat([e1_emb, e2_emb], dim=-1))

        loss = self.loss_fn(logits, labels) if labels is not None else None
        return loss, logits, e1_emb, e2_emb


# ══════════════════════════════════════════════════════════════
# Stage 2: HGNN — 仿 DualHGCN 重写
# ══════════════════════════════════════════════════════════════

def generate_G_from_H(H, W=None, conv='sym'):
    """
    仿 DualHGCN.generate_G_from_H:
    将 incidence matrix H (N×M) 转换为传播矩阵 G (N×N)

    sym:  G = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
    asym: G = D_v^{-1}   H W D_e^{-1} H^T

    H: torch.Tensor (N, M), N=nodes, M=hyperedges
    返回: G torch.Tensor (N, N)
    """
    # 转 numpy 计算（DualHGCN 原始实现用 numpy，数值更稳定）
    H_np = H.detach().cpu().numpy()
    n_edge = H_np.shape[1]

    if W is None:
        W = np.ones(n_edge)
    else:
        W = np.asarray(W, dtype=np.float32)
        if W.shape[0] != n_edge:
            raise ValueError(f'edge weight size mismatch: {W.shape[0]} vs {n_edge}')
    DV = np.sum(H_np * W, axis=1)   # node degree, shape (N,)
    DE = np.sum(H_np, axis=0)       # edge degree, shape (M,)

    # 防止除零
    DV = np.clip(DV, 1e-12, None)
    DE = np.clip(DE, 1e-12, None)

    invDE = np.diag(DE ** -1)
    W_mat = np.diag(W)
    H_mat = H_np
    HT = H_mat.T

    if conv == 'sym':
        DV_inv_sqrt = np.diag(DV ** -0.5)
        G = DV_inv_sqrt @ H_mat @ W_mat @ invDE @ HT @ DV_inv_sqrt
    else:  # asym
        DV_inv = np.diag(DV ** -1)
        G = DV_inv @ H_mat @ W_mat @ invDE @ HT

    return torch.from_numpy(G).float()


def build_incidence_matrix(n_events, causal_pairs, device,
                           edge_dropout=0.0, training=False,
                           neighbor_probs=None):
    """
    构建 incidence matrix H (N × M):
    【修改】拆分为有向字典，矩阵列数翻倍 (M=2N)。前N列为cause边，后N列为effect边。
    """
    # Step 1: 拆分为独立的因果字典
    effects_of = {i: set() for i in range(n_events)}
    causes_of = {i: set() for i in range(n_events)}
    
    for u, v in causal_pairs:
        # 严格按照有向元组方向 (u: cause, v: effect)
        effects_of[u].add(v)
        causes_of[v].add(u)

    # Step 2: 构建 hyperedges (分为前N个cause边和后N个effect边)
    hyperedges = []
    edge_weights = []
    
    # 2.1 Cause-hyperedges (前 N 个，中心节点是 i)
    for m in range(n_events):
        he = {m}
        he.update(effects_of[m])
        hyperedges.append(he)
        if neighbor_probs:
            probs = [neighbor_probs.get((m, v), 0.0) for v in effects_of[m]]
            w = float(np.mean(probs)) if probs else 1.0
            edge_weights.append(w)
        else:
            edge_weights.append(1.0)
        
    # 2.2 Effect-hyperedges (后 N 个，中心节点是 i)
    for m in range(n_events):
        he = {m}
        he.update(causes_of[m])
        hyperedges.append(he)
        if neighbor_probs:
            probs = [neighbor_probs.get((u, m), 0.0) for u in causes_of[m]]
            w = float(np.mean(probs)) if probs else 1.0
            edge_weights.append(w)
        else:
            edge_weights.append(1.0)

    # Step 3: 训练时 edge dropout
    if training and edge_dropout > 0:
        import random
        new_hyperedges = []
        new_weights = []
        for col, he in enumerate(hyperedges):
            centre = col % n_events  # 获取真实的中心节点索引
            others = he - {centre}
            kept = {centre}
            for node in others:
                if random.random() > edge_dropout:
                    kept.add(node)
            new_hyperedges.append(kept)
            new_weights.append(edge_weights[col])
        hyperedges = new_hyperedges
        edge_weights = new_weights

    # Step 4: 转换为 incidence matrix
    M = len(hyperedges)  # 此时 M = 2 * N
    H = torch.zeros(n_events, M, device=device)
    for col, he in enumerate(hyperedges):
        for node in he:
            H[node, col] = 1.0

    return H, edge_weights


class HGNNLayer(nn.Module):
    """
    仿 DualHGCN 的 HyperConv:
    输入: node features E (N, d_in), 预计算的传播矩阵 G (N, N)
    输出: E' = σ(G · E · Θ) + E  (带 residual)

    关键区别 vs v1:
      - G 是预计算的，不在 forward 里算 D^{-1/2} A ... 这些
      - 有 residual connection
      - 更简单、更稳定
    """
    def __init__(self, in_dim, out_dim, dropout=0.1, residual=True):
        super().__init__()
        self.theta   = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.ReLU()
        self.residual = residual

        # 如果维度不匹配，用线性投影做 residual
        if residual and in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.res_proj = None

    def forward(self, E, G):
        """
        E: (N, d_in) node features
        G: (N, N) pre-computed propagation matrix
        """
        # 线性变换
        E_proj = self.theta(E)           # (N, d_out)
        # 图传播: G · E_proj
        E_conv = G @ E_proj              # (N, d_out)
        E_conv = self.act(E_conv)
        E_conv = self.dropout(E_conv)

        # Residual
        if self.residual:
            if self.res_proj is not None:
                E_conv = E_conv + self.res_proj(E)
            else:
                E_conv = E_conv + E

        return E_conv


class HGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden_dim
            self.layers.append(
                HGNNLayer(d_in, hidden_dim, dropout, residual=True)
            )

    def forward(self, E, G):
        """
        E: (N, d) initial node embeddings
        G: (N, N) pre-computed propagation matrix (from generate_G_from_H)
        """
        for layer in self.layers:
            E = layer(E, G)
        return E


# ── Stage 2 Full Classifier ───────────────────────────────────
class EHNEMClassifier(nn.Module):
    """
    论文 Eq.8: v_k = [e_i_ESE || e_j_ESE || e_i_HGNN || e_j_HGNN]  (4d)

    改进:
    1. G 预计算 (generate_G_from_H)，不再在 HGNN forward 里做
    2. Gating: 学习 ESE vs HGNN 特征的权重
       gate = σ(W_g · [ese_feat || hgnn_feat])
       final = gate * hgnn_feat + (1-gate) * ese_feat
       → 如果 HGNN 输出质量差，gate 会学到偏向 ESE
    3. 支持 conv='sym'/'asym' (DualHGCN 的 asym 通常更好)
    """
    def __init__(self, args):
        super().__init__()
        d = 768
        self.d = d

        self.hgnn = HGNN(
            d, d,
            num_layers=args.hgnn_layers,
            dropout=args.dropout_hgnn,
        )

        # Gating mechanism: 让模型学习 ESE vs HGNN 的信任度
        self.gate = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.Sigmoid(),
        )

        # 最终分类器: 输入维度 = d * 4
        # [gated_e1 || gated_e2 || ese_e1 || ese_e2]
        self.classifier = nn.Linear(d * 4, 2)
        self.loss_fn    = _build_loss_fn(args)

        self.conv_type     = getattr(args, 'hgnn_conv', 'sym')
        self.edge_dropout  = getattr(args, 'edge_dropout', 0.3)

    def forward(self, event_embs, causal_pairs, pair_indices,
                labels=None, training=False, neighbor_probs=None):
        """
        event_embs:   (N, d) ESE 输出的 event embeddings
        causal_pairs: list[(i, j)] 用于构建超图的因果对
        pair_indices: list[(i, j)] 所有需要分类的 event pair
        labels:       (P,) 标签
        training:     是否训练模式（控制 edge_dropout）
        """
        device = event_embs.device
        N      = event_embs.size(0)

        # ── Step 1: 构建 incidence matrix H ──
        H, W = build_incidence_matrix(
            N, causal_pairs, device,
            edge_dropout=self.edge_dropout if training else 0.0,
            training=training,
            neighbor_probs=neighbor_probs if not training else None,
        )

        # ── Step 2: 预计算传播矩阵 G（仿 DualHGCN）──
        G = generate_G_from_H(H, W=W, conv=self.conv_type).to(device)

        # ── Step 3: HGNN 传播 ──
        hgnn_emb = self.hgnn(event_embs, G)   # (N, d)

        if not pair_indices:
            return None, torch.zeros(0, 2, device=device)

        idx_i = torch.tensor([p[0] for p in pair_indices], device=device)
        idx_j = torch.tensor([p[1] for p in pair_indices], device=device)

        # ESE embeddings for pairs
        ese_ei = event_embs[idx_i]     # (P, d)
        ese_ej = event_embs[idx_j]     # (P, d)

        # HGNN embeddings for pairs
        hgnn_ei = hgnn_emb[idx_i]      # (P, d)
        hgnn_ej = hgnn_emb[idx_j]      # (P, d)

        # ── Step 4: Gating — 让模型决定多大程度信任 HGNN ──
        gate_i = self.gate(torch.cat([ese_ei, hgnn_ei], dim=-1))   # (P, d)
        gate_j = self.gate(torch.cat([ese_ej, hgnn_ej], dim=-1))   # (P, d)

        fused_ei = gate_i * hgnn_ei + (1 - gate_i) * ese_ei   # (P, d)
        fused_ej = gate_j * hgnn_ej + (1 - gate_j) * ese_ej   # (P, d)

        # ── Step 5: 分类 ──
        # 论文 Eq.8 拼接: [ese || ese || hgnn_fused || hgnn_fused]
        v_k = torch.cat([ese_ei, ese_ej, fused_ei, fused_ej], dim=-1)  # (P, 4d)
        logits = self.classifier(v_k)   # (P, 2)

        loss = self.loss_fn(logits, labels) if labels is not None else None
        return loss, logits
