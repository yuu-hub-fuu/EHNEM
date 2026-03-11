"""
train_ehnem.py — EHNEM 두 단계 훈련 (修复版)

修复点 (vs v1):
1. HGNN 训练时用 GT causal pairs 构建超图 (论文 §3.2.1 明确要求)
   + edge_dropout 缩小 train/test 分布差距
2. HGNN 推理时用 ESE predicted pairs (论文 §3.2.1)
3. 传 training flag 给 classifier 控制 edge_dropout
4. 新增 args: edge_dropout, hgnn_conv
"""

import os, os.path as osp, argparse, logging, pickle, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold

from model_ehnem import ESEModel, EHNEMClassifier
from processor_xml import XMLProcessor, DocDataset

import sys
sys.path.insert(0, osp.dirname(__file__))
try:
    from utils import set_seed, compute_f1
except ImportError:
    def set_seed(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    def compute_f1(gold, predicted, logger=None):
        cc = cp = cg = 0
        for g, p in zip(gold, predicted):
            if g: cg += 1
            if p: cp += 1
            if g and p: cc += 1
        pr = cc / (cp + 1e-10)
        rc = cc / (cg + 1e-10)
        f1 = 2*pr*rc / (pr+rc+1e-10) if pr+rc > 1e-4 else 0.
        if logger:
            logger.info(f'correct {cc}, predicted {cp}, golden {cg}')
        return pr, rc, f1

logger = logging.getLogger(__name__)


# ================================================================
# 공통: chunk 단위 ESE forward (OOM 방지)
# ================================================================

def _ese_forward_chunks(args, model, pfs, with_grad=True):
    chunk_size = getattr(args, 'chunk_size', 4)
    all_logits, all_labels = [], []
    total_loss = 0.
    n_chunks   = 0

    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        for start in range(0, len(pfs), chunk_size):
            chunk   = pfs[start: start + chunk_size]
            max_len = max(len(pf.enc_input_ids) for pf in chunk)
            B       = len(chunk)

            input_ids = torch.ones(B, max_len, dtype=torch.long, device=args.device)
            mask_ids  = torch.zeros(B, max_len, dtype=torch.long, device=args.device)
            for i, pf in enumerate(chunk):
                L = len(pf.enc_input_ids)
                input_ids[i, :L] = torch.tensor(pf.enc_input_ids, device=args.device)
                mask_ids[i, :L]  = 1

            e1_spans = [(pf.e1_start, pf.e1_end) for pf in chunk]
            e2_spans = [(pf.e2_start, pf.e2_end) for pf in chunk]
            labels_t = torch.tensor([pf.label for pf in chunk],
                                    dtype=torch.long, device=args.device)

            loss, logits, _, _ = model(input_ids, mask_ids,
                                       e1_spans, e2_spans, labels_t)

            if with_grad and loss is not None:
                loss.backward()

            total_loss += loss.item() if loss is not None else 0.
            n_chunks   += 1
            all_logits.append(logits.detach().cpu())
            all_labels += labels_t.cpu().tolist()

            del input_ids, mask_ids, labels_t, loss, logits
            torch.cuda.empty_cache()

    if n_chunks == 0:
        return None, None, None

    return (total_loss / n_chunks,
            torch.cat(all_logits, dim=0),
            all_labels)


# ================================================================
# Stage 1: ESE (不变)
# ================================================================

def ese_train_epoch(args, model, docs, optimizer, scheduler):
    model.train()
    total_loss, all_gold, all_pred = [], [], []

    for step, doc in enumerate(docs):
        if not doc.pair_features:
            continue
        optimizer.zero_grad()
        loss_val, logits, labels = _ese_forward_chunks(
            args, model, doc.pair_features, with_grad=True)
        if loss_val is None:
            continue
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step(); scheduler.step()
        total_loss.append(loss_val)
        all_pred += torch.argmax(logits, -1).tolist()
        all_gold += labels

        if step % args.logging_steps == 0:
            p, r, f1 = compute_f1(all_gold, all_pred, logger)
            logger.info(f'  Step {step}: loss={np.mean(total_loss):.4f} '
                        f'P={p:.4f} R={r:.4f} F1={f1:.4f}')

    p, r, f1 = compute_f1(all_gold, all_pred, logger)
    return np.mean(total_loss) if total_loss else 0., p, r, f1


def ese_eval_epoch(args, model, docs):
    model.eval()
    total_loss, all_gold, all_pred = [], [], []
    prob_dict = {}

    for doc in docs:
        if not doc.pair_features:
            continue
        loss_val, logits, labels = _ese_forward_chunks(
            args, model, doc.pair_features, with_grad=False)
        if loss_val is None:
            continue
        total_loss.append(loss_val)
        probs = torch.softmax(logits, -1)[:, 1].tolist()
        preds = torch.argmax(logits, -1).tolist()
        all_pred += preds
        all_gold += labels

        for pf, prob in zip(doc.pair_features, probs):
            prob_dict[(doc.doc_id, pf.event_key1, pf.event_key2)] = prob

    p, r, f1 = compute_f1(all_gold, all_pred, logger)
    return np.mean(total_loss) if total_loss else 0., p, r, f1, prob_dict


# ================================================================
# Stage 2: HGNN (修复版)
# ================================================================

def _get_event_embs(args, ese_model, doc):
    """chunk 단위로 ESE 실행, event 별 평균 embedding 반환"""
    chunk_size = getattr(args, 'chunk_size', 4)
    accum = {}

    with torch.no_grad():
        for start in range(0, len(doc.pair_features), chunk_size):
            chunk   = doc.pair_features[start: start + chunk_size]
            max_len = max(len(pf.enc_input_ids) for pf in chunk)
            B       = len(chunk)
            input_ids = torch.ones(B, max_len, dtype=torch.long, device=args.device)
            mask_ids  = torch.zeros(B, max_len, dtype=torch.long, device=args.device)
            for i, pf in enumerate(chunk):
                L = len(pf.enc_input_ids)
                input_ids[i, :L] = torch.tensor(pf.enc_input_ids, device=args.device)
                mask_ids[i, :L]  = 1

            e1_spans = [(pf.e1_start, pf.e1_end) for pf in chunk]
            e2_spans = [(pf.e2_start, pf.e2_end) for pf in chunk]
            _, _, e1_emb, e2_emb = ese_model(input_ids, mask_ids,
                                             e1_spans, e2_spans)
            for i, pf in enumerate(chunk):
                for key, emb in [(pf.event_key1, e1_emb[i]),
                                 (pf.event_key2, e2_emb[i])]:
                    accum.setdefault(key, []).append(emb.detach().cpu())

            del input_ids, mask_ids, e1_emb, e2_emb
            torch.cuda.empty_cache()

    return {k: torch.stack(vs).mean(0) for k, vs in accum.items()}


def _hgnn_forward_doc(args, ese_model, hgnn_model, doc, causal_probs,
                       training=False):
    """
    修复版: 训练/推理用不同的超图构建策略

    训练 (training=True):
      用 GT causal pairs 构建超图 (论文 §3.2.1)
      + edge_dropout 模拟推理时的噪声

    推理 (training=False):
      用 ESE predicted probs 构建超图 (论文 §3.2.1)
    """
    eid2emb    = _get_event_embs(args, ese_model, doc)
    valid_keys = [k for k in doc.event_keys if k in eid2emb]
    if len(valid_keys) < 2:
        return None, None, None

    eid2idx    = {k: i for i, k in enumerate(valid_keys)}
    event_embs = torch.stack([eid2emb[k] for k in valid_keys]).to(args.device)

    # ── 构建超图的因果对 ──
    if training:
        # 训练: 用 GT causal pairs (论文明确要求)
        # 【修改】使用有向的 doc.gt_causal_pairs，确保提取严格的 (因, 果) 方向
        causal_pairs = []
        for c1, c2 in doc.gt_causal_pairs:
            if c1 in eid2idx and c2 in eid2idx:
                causal_pairs.append((eid2idx[c1], eid2idx[c2]))
    else:
        # 推理: 用 ESE predicted probs 构建超图
        causal_pairs = []
        neighbor_probs = {}
        for pf in doc.pair_features:
            k1, k2 = pf.event_key1, pf.event_key2
            if k1 not in eid2idx or k2 not in eid2idx:
                continue
            prob = causal_probs.get((doc.doc_id, k1, k2),
                   causal_probs.get((doc.doc_id, k2, k1), 0.))
            if prob >= args.causal_threshold:
                causal_pairs.append((eid2idx[k1], eid2idx[k2]))
                neighbor_probs[(eid2idx[k1], eid2idx[k2])] = prob

    # ── 要分类的所有 pair ──
    pair_indices, labels = [], []
    for pf in doc.pair_features:
        k1, k2 = pf.event_key1, pf.event_key2
        if k1 not in eid2idx or k2 not in eid2idx:
            continue
        pair_indices.append((eid2idx[k1], eid2idx[k2]))
        labels.append(pf.label)

    if not pair_indices:
        return None, None, None

    labels_t = torch.tensor(labels, dtype=torch.long, device=args.device)
    loss, logits = hgnn_model(
        event_embs, causal_pairs, pair_indices, labels_t,
        training=training,
        neighbor_probs=neighbor_probs if not training else None,
    )
    return loss, logits, labels


def hgnn_train_epoch(args, ese_model, hgnn_model, docs,
                     optimizer, scheduler, causal_probs):
    hgnn_model.train(); ese_model.eval()
    total_loss, all_gold, all_pred = [], [], []

    for step, doc in enumerate(docs):
        if not doc.pair_features:
            continue
        optimizer.zero_grad()
        loss, logits, labels = _hgnn_forward_doc(
            args, ese_model, hgnn_model, doc, causal_probs,
            training=True)   # ← 训练模式: GT + edge_dropout
        if loss is None:
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(hgnn_model.parameters(), args.max_grad_norm)
        optimizer.step(); scheduler.step()

        total_loss.append(loss.item())
        all_pred += torch.argmax(logits.detach(), -1).cpu().tolist()
        all_gold += labels

    p, r, f1 = compute_f1(all_gold, all_pred, logger)
    return np.mean(total_loss) if total_loss else 0., p, r, f1


def hgnn_eval_epoch(args, ese_model, hgnn_model, docs, causal_probs):
    hgnn_model.eval(); ese_model.eval()
    total_loss, all_gold, all_pred = [], [], []

    with torch.no_grad():
        for doc in docs:
            if not doc.pair_features:
                continue
            loss, logits, labels = _hgnn_forward_doc(
                args, ese_model, hgnn_model, doc, causal_probs,
                training=False)   # ← 推理模式: predicted + no dropout
            if loss is None:
                continue
            total_loss.append(loss.item())
            all_pred += torch.argmax(logits, -1).cpu().tolist()
            all_gold += labels

    p, r, f1 = compute_f1(all_gold, all_pred, logger)
    return np.mean(total_loss) if total_loss else 0., p, r, f1


# ================================================================
# Fold 단위 훈련
# ================================================================

# ================================================================
# Fold 단위 훈련
# ================================================================

def run_fold(args, fold, tokenizer, train_doc_ds, dev_doc_ds,
             train_indices, test_indices):
    train_docs = [train_doc_ds.docs[i] for i in train_indices]
    test_docs  = [train_doc_ds.docs[i] for i in test_indices]
    dev_docs   = dev_doc_ds.docs

    no_decay = ['bias', 'LayerNorm.weight']

    # ── Stage 1: ESE ────────────────────────────────────────
    logger.info(f"--- Fold {fold} Stage 1: ESE ---")
    ese_model = ESEModel(args).to(args.device)
    ese_model.roberta.resize_token_embeddings(len(tokenizer))

    optimizer = AdamW([
        {'params': [p for n,p in ese_model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n,p in ese_model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ], lr=args.learning_rate)

    t_total = len(train_docs) * args.ese_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total * args.warmup_ratio),
        num_training_steps=t_total
    )

    best_dev_f1 = 0.
    for epoch in range(args.ese_epochs):
        tr_loss, tr_p, tr_r, tr_f1 = ese_train_epoch(
            args, ese_model, train_docs, optimizer, scheduler)
        _, dev_p, dev_r, dev_f1, _ = ese_eval_epoch(args, ese_model, dev_docs)
        logger.info(f"[ESE] Epoch {epoch+1}/{args.ese_epochs} | "
                    f"Train Loss: {tr_loss:.4f} F1: {tr_f1:.4f} | Dev F1: {dev_f1:.4f}")

        if dev_f1 >= best_dev_f1:
            best_dev_f1 = dev_f1
            torch.save(ese_model.state_dict(),
                       osp.join(args.output_dir, f'ese_fold{fold}.pt'))

    # Load best ESE
    ese_model.load_state_dict(torch.load(osp.join(args.output_dir, f'ese_fold{fold}.pt')))
    
    # 【修复点 1】：分离 Test 和 Dev 的预测概率字典
    _, test_p, test_r, test_f1, causal_probs_test = ese_eval_epoch(
        args, ese_model, test_docs)
    logger.info(f"[ESE] Fold {fold} Test | P: {test_p:.4f}  R: {test_r:.4f}  F1: {test_f1:.4f}")

    # 获取验证集的真实预测概率，不再是空图
    _, _, _, _, causal_probs_dev = ese_eval_epoch(args, ese_model, dev_docs)

    # Stage 1 预测提取完毕，冻结 ESE 准备 Stage 2
    for param in ese_model.parameters():
        param.requires_grad = False

    # ── Stage 2: HGNN ───────────────────────────────────────
    logger.info(f"--- Fold {fold} Stage 2: HGNN ---")
    hgnn_model = EHNEMClassifier(args).to(args.device)

    opt_hgnn = AdamW(hgnn_model.parameters(), lr=args.hgnn_lr, weight_decay=args.weight_decay)
    t_total_hgnn = len(train_docs) * args.hgnn_epochs
    sched_hgnn = get_linear_schedule_with_warmup(
        opt_hgnn, num_warmup_steps=int(t_total_hgnn * args.warmup_ratio),
        num_training_steps=t_total_hgnn
    )

    best_hgnn_f1 = 0.
    for epoch in range(args.hgnn_epochs):
        # 训练过程传入 {} 是没问题的，因为训练用的是 GT (真实标签) 建图
        tr_loss, tr_p, tr_r, tr_f1 = hgnn_train_epoch(
            args, ese_model, hgnn_model, train_docs,
            opt_hgnn, sched_hgnn, causal_probs={}) 
        
        # 【修复点 2】：验证过程传入 causal_probs_dev，触发超图动态构建
        _, dev_p, dev_r, dev_f1 = hgnn_eval_epoch(
            args, ese_model, hgnn_model, dev_docs, causal_probs_dev)

        logger.info(f"[HGNN] Epoch {epoch+1}/{args.hgnn_epochs} | "
                    f"Train Loss: {tr_loss:.4f} F1: {tr_f1:.4f} | Dev F1: {dev_f1:.4f}")

        if dev_f1 >= best_hgnn_f1:
            best_hgnn_f1 = dev_f1
            torch.save(hgnn_model.state_dict(),
                       osp.join(args.output_dir, f'hgnn_fold{fold}.pt'))

    # Load best HGNN
    hgnn_model.load_state_dict(torch.load(osp.join(args.output_dir, f'hgnn_fold{fold}.pt')))
    
    # 【修复点 3】：Test 评估阶段同样使用独立的 causal_probs_test
    _, final_p, final_r, final_f1 = hgnn_eval_epoch(
        args, ese_model, hgnn_model, test_docs, causal_probs_test)
    
    logger.info(f"[HGNN] Fold {fold} Final Test | P: {final_p:.4f}  R: {final_r:.4f}  F1: {final_f1:.4f}")
    
    return final_p, final_r, final_f1


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./v0.9')
    parser.add_argument('--output_dir', default='./outputs')
    parser.add_argument('--model_name_or_path', default='roberta-base')
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--k_fold', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chunk_size', type=int, default=8, help='ESE forward batch size (OOM 방지)')
    
    # Model args
    parser.add_argument('--dropout_ese', type=float, default=0.5)
    parser.add_argument('--dropout_hgnn', type=float, default=0.1)
    parser.add_argument('--edge_dropout', type=float, default=0.3, help='HGNN 훈련 시 초과 간선 무작위 제거 비율')
    parser.add_argument('--hgnn_layers', type=int, default=2)
    parser.add_argument('--hgnn_conv', type=str, default='sym', choices=['sym', 'asym'])
    parser.add_argument('--loss_type', default='ce', choices=['ce', 'focal'])
    parser.add_argument('--class_weight', type=float, default=0.75, help='Pos class weight (CE)')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma')
    
    # Training args
    parser.add_argument('--ese_epochs', type=int, default=15)
    parser.add_argument('--hgnn_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--hgnn_lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--causal_threshold', type=float, default=0.5, help='ESE prediction threshold')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Logger 설정
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[logging.FileHandler(osp.join(args.output_dir, 'train.log')),
                  logging.StreamHandler()]
    )
    set_seed(args.seed)
    
    logger.info(f"Args: {args}")

    # 데이터 로드/캐싱
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name_or_path)
    tokenizer.add_tokens(['<t>', '</t>'])
    
    cache_path = osp.join(args.output_dir, 'doc_dataset.pkl')
    if osp.exists(cache_path):
        logger.info(f"Loading cached dataset from {cache_path}")
        with open(cache_path, 'rb') as f:
            all_doc_ds = pickle.load(f)
    else:
        logger.info("Building dataset from XML...")
        processor = XMLProcessor(args, tokenizer)
        all_doc_ds = processor.build_doc_dataset(args.data_dir)
        with open(cache_path, 'wb') as f:
            pickle.dump(all_doc_ds, f)
            
    # Topic 기준 K-Fold Split
    topics = set()
    for doc in all_doc_ds.docs:
        topics.add(doc.topic_id)
    unique_topics = sorted(list(topics))
    topics_ordered = [doc.topic_id for doc in all_doc_ds.docs]

    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    all_p, all_r, all_f1 = [], [], []

    for fold, (topic_train_idx, topic_test_idx) in enumerate(
            kf.split(unique_topics), start=1):
        train_topics_list = [unique_topics[i] for i in topic_train_idx]
        test_topics  = {unique_topics[i] for i in topic_test_idx}

        # Dev set logic: 划分 train_topics 的最后 1 个 topic 作为 validation
        dev_topics = {train_topics_list[-1]}
        train_topics = set(train_topics_list[:-1])

        train_idx = [i for i, t in enumerate(topics_ordered) if t in train_topics]
        dev_idx   = [i for i, t in enumerate(topics_ordered) if t in dev_topics]
        test_idx  = [i for i, t in enumerate(topics_ordered) if t in test_topics]

        dev_doc_ds = DocDataset([all_doc_ds.docs[i] for i in dev_idx])

        logger.info(f"\n{'='*60}\nFold {fold}/{args.k_fold} "
                    f"train_topics={sorted(train_topics)} "
                    f"dev_topics={sorted(dev_topics)} "
                    f"test_topics={sorted(test_topics)}\n{'='*60}")
        
        p, r, f1 = run_fold(args, fold, tokenizer, all_doc_ds, dev_doc_ds,
                            train_idx, test_idx)
        all_p.append(p); all_r.append(r); all_f1.append(f1)

    logger.info(f"\n{'='*60}\n[K-Fold Final Results]")
    logger.info(f"Precision: {np.mean(all_p):.4f} ± {np.std(all_p):.4f}")
    logger.info(f"Recall:    {np.mean(all_r):.4f} ± {np.std(all_r):.4f}")
    logger.info(f"F1 Score:  {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")

if __name__ == '__main__':
    main()
