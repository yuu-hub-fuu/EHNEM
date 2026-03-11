"""
processor_xml.py — 直接从 ESC v0.9 XML 构建 EHNEM 训练数据

彻底绕开 ERGO pkl，从源数据出发：
1. 解析 ESC XML: tokens, events, PLOT_LINK (causal relations)
2. 构建 event pairs (同文档内所有 event 两两配对)
3. 用 RoBERTa 编码，同句双标记 / 跨句 sentence-pair
4. 输出 DocFeature 供 train_ehnem.py 使用

因果关系判断 (ESC README):
  PLOT_LINK 的 CAUSES="TRUE"  → source causes target
  PLOT_LINK 的 CAUSED_BY="TRUE" → target causes source
"""

import os, logging, glob
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Set
from lxml import etree
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ── 数据容器 (与 processor_ehnem.py 兼容) ─────────────────────
class PairFeature:
    def __init__(self, doc_id, k1, k2, input_ids, mask_ids,
                 e1_start, e1_end, e2_start, e2_end, label):
        self.doc_id        = doc_id
        self.event_key1    = str(k1)
        self.event_key2    = str(k2)
        self.enc_input_ids = input_ids
        self.enc_mask_ids  = mask_ids
        self.e1_start      = e1_start
        self.e1_end        = e1_end
        self.e2_start      = e2_start
        self.e2_end        = e2_end
        self.label         = int(label)

class DocFeature:
    def __init__(self, doc_id, topic_id, event_keys, pair_features, gt_causal_pairs):
        self.doc_id          = doc_id
        self.topic_id        = topic_id
        self.event_keys      = event_keys
        self.pair_features   = pair_features
        self.gt_causal_pairs = gt_causal_pairs

class DocDataset(Dataset):
    def __init__(self, docs): self.docs = docs
    def __len__(self): return len(self.docs)
    def __getitem__(self, idx): return self.docs[idx]


# ── XML 解析 ─────────────────────────────────────────────────
class ESCDocument:
    """解析单个 ESC XML 文件"""
    def __init__(self, xml_path, topic_id):
        self.path = xml_path
        self.topic_id = str(topic_id)
        self.doc_id = os.path.basename(xml_path).replace('.xml.xml', '').replace('.xml', '')

        self.tokens = {}       # t_id(int) → {word, sentence(int), number(int)}
        self.events = {}       # m_id(str) → {type, token_ids: [t_id, ...]}
        self.causal_pairs = [] # [(cause_m_id, effect_m_id), ...]

        self._parse()

    def _parse(self):
        tree = etree.parse(self.path)
        root = tree.getroot()

        # 1. 解析 tokens
        for tok_elem in root.iter('token'):
            t_id = int(tok_elem.get('t_id'))
            self.tokens[t_id] = {
                'word': tok_elem.text or '',
                'sentence': int(tok_elem.get('sentence')),
                'number': int(tok_elem.get('number')),
            }

        # 2. 解析 events (ACTION_* 和 NEG_ACTION_*)
        markables = root.find('.//Markables')
        if markables is not None:
            for elem in markables:
                tag = elem.tag
                if 'ACTION' not in tag:
                    continue
                m_id = elem.get('m_id')
                if m_id is None:
                    continue
                # 收集 token_anchor
                token_ids = []
                for anchor in elem.findall('token_anchor'):
                    tid = anchor.get('t_id')
                    if tid:
                        token_ids.append(int(tid))
                if token_ids:
                    self.events[m_id] = {
                        'type': tag,
                        'token_ids': sorted(token_ids),
                    }

        # 3. 解析 PLOT_LINK (因果关系)
        self._plot_link_total = 0
        self._plot_link_skipped = 0
        relations = root.find('.//Relations')
        if relations is not None:
            for link in relations.findall('PLOT_LINK'):
                rel_type = (link.get('relType') or '').strip().upper()
                if not rel_type:
                    continue

                # 新增：仅放行这两种具有明确因果方向的类型
                if rel_type not in ['PRECONDITION', 'FALLING_ACTION']:
                    continue

                self._plot_link_total += 1

                source_elem = link.find('source')
                target_elem = link.find('target')
                if source_elem is None or target_elem is None:
                    continue
                source_mid = source_elem.get('m_id')
                target_mid = target_elem.get('m_id')
                if source_mid is None or target_mid is None:
                    continue

                if source_mid not in self.events or target_mid not in self.events:
                    self._plot_link_skipped += 1
                    continue

                # 替换：基于类型判断真实的因果方向并有向追加，彻底弃用双向 append
                if rel_type == 'PRECONDITION':
                    self.causal_pairs.append((source_mid, target_mid))
                elif rel_type == 'FALLING_ACTION':
                    self.causal_pairs.append((target_mid, source_mid))

    def get_sentences(self) -> Dict[int, List[Tuple[int, int, str]]]:
        """返回 sent_id → [(number, t_id, word), ...] 按 number 排序"""
        sents = defaultdict(list)
        for t_id, info in self.tokens.items():
            sents[info['sentence']].append((info['number'], t_id, info['word']))
        for sid in sents:
            sents[sid].sort(key=lambda x: x[0])
        return dict(sents)

    def get_event_info(self) -> Dict[str, Tuple[int, Set[int]]]:
        """返回 m_id → (sent_id, {number_in_sent, ...})"""
        result = {}
        for m_id, ev in self.events.items():
            tids = ev['token_ids']
            # 所有 token 应该在同一个句子
            sent_ids = set()
            numbers = set()
            for tid in tids:
                tok = self.tokens.get(tid)
                if tok:
                    sent_ids.add(tok['sentence'])
                    numbers.add(tok['number'])
            if sent_ids:
                # 取第一个 token 的 sentence (多词 event 应同句)
                sent_id = self.tokens[tids[0]]['sentence']
                result[m_id] = (sent_id, numbers)
        return result

    def get_causal_set(self) -> Set[Tuple[str, str]]:
        """返回所有因果对的 set，用于快速查找"""
        return set(self.causal_pairs)


# ── Processor ────────────────────────────────────────────────
class XMLProcessor:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tok = tokenizer
        self.max_len = getattr(args, 'max_seq_len', 256)
        self.t_open = tokenizer.convert_tokens_to_ids('<t>')
        self.t_close = tokenizer.convert_tokens_to_ids('</t>')
        # 统计
        self.span_hit = 0
        self.span_miss = 0
        self.total_pos = 0
        self.total_neg = 0
        self.total_pairs = 0
        self.skipped_sent0 = 0

    def build_all_docs(self, esc_root) -> List[DocFeature]:
        """
        esc_root: ESC v0.9 根目录，下面有 topic 文件夹
        结构: esc_root/{topic_id}/{doc_files}.xml.xml
        """
        docs = []
        topic_dirs = sorted([
            d for d in os.listdir(esc_root)
            if os.path.isdir(os.path.join(esc_root, d))
        ])

        for topic_id in topic_dirs:
            topic_path = os.path.join(esc_root, topic_id)
            xml_files = sorted(glob.glob(os.path.join(topic_path, '*ecbplus*')))

            for xml_path in xml_files:
                try:
                    esc_doc = ESCDocument(xml_path, topic_id)
                except Exception as e:
                    logger.warning(f"Failed to parse {xml_path}: {e}")
                    continue

                df = self._build_doc_feature(esc_doc)
                if df:
                    docs.append(df)

        self._report_stats()
        return docs

    def build_doc_dataset(self, esc_root) -> DocDataset:
        docs = self.build_all_docs(esc_root)
        logger.info(f"Built {len(docs)} docs from XML")
        return DocDataset(docs)

    def _report_stats(self):
        total = self.span_hit + self.span_miss
        logger.info(
            f"[XML Stats] docs processed, pairs={self.total_pairs} "
            f"pos={self.total_pos} neg={self.total_neg} "
            f"pos_ratio={self.total_pos/max(self.total_pairs,1)*100:.1f}%"
        )
        if total > 0:
            logger.info(
                f"[Span stats] hit={self.span_hit} miss={self.span_miss} "
                f"miss_rate={self.span_miss/max(total,1)*100:.1f}%"
            )
        if self.skipped_sent0:
            logger.info(f"[Filter] sent_id=0 skipped: {self.skipped_sent0}")

    def _build_doc_feature(self, esc_doc: ESCDocument) -> Optional[DocFeature]:
        """从解析好的 ESCDocument 构建 DocFeature"""
        sentences = esc_doc.get_sentences()
        event_info = esc_doc.get_event_info()  # m_id → (sent_id, {numbers})
        causal_set = esc_doc.get_causal_set()

        if len(event_info) < 2:
            return None

        # 所有 event m_id 列表
        event_mids = sorted(event_info.keys())

        # 构建句子文本缓存: sent_id → [(number, word), ...]
        sent_words = {}
        for sid, toks in sentences.items():
            sent_words[sid] = [(num, word) for num, tid, word in toks]

        # ── 标记函数 ──────────────────────────────────────────
        def build_marked_sent_single(sid, pos_set):
            """单 event 标记"""
            words = sorted(sent_words.get(sid, []), key=lambda x: x[0])
            if not words:
                return None, False
            result, in_span, found = [], False, False
            for pos, word in words:
                if pos in pos_set:
                    if not in_span:
                        result.append('<t>'); in_span = True
                    result.append(word); found = True
                else:
                    if in_span:
                        result.append('</t>'); in_span = False
                    result.append(word)
            if in_span:
                result.append('</t>')
            return ' '.join(result), found

        def build_marked_sent_dual(sid, pos_set1, pos_set2):
            """同句双 event 标记"""
            words = sorted(sent_words.get(sid, []), key=lambda x: x[0])
            if not words:
                return None, False, False, True
            min1 = min(pos_set1) if pos_set1 else float('inf')
            min2 = min(pos_set2) if pos_set2 else float('inf')
            e1_first = (min1 <= min2)
            result, current_span, found1, found2 = [], 0, False, False
            for pos, word in words:
                if pos in pos_set1: tag = 1
                elif pos in pos_set2: tag = 2
                else: tag = 0
                if current_span != 0 and tag != current_span:
                    result.append('</t>'); current_span = 0
                if tag != 0 and current_span == 0:
                    result.append('<t>'); current_span = tag
                    if tag == 1: found1 = True
                    if tag == 2: found2 = True
                result.append(word)
            if current_span != 0:
                result.append('</t>')
            return ' '.join(result), found1, found2, e1_first

        # ── 构建所有 event pair ───────────────────────────────
        pair_features = []
        gt_causal = []

        for i, mid1 in enumerate(event_mids):
            for j, mid2 in enumerate(event_mids):
                # 替换：允许全排列双向遍历，仅排除自己和自己配对，赋能方向感知
                if i == j:  
                    continue

                sid1, pos_set1 = event_info[mid1]
                sid2, pos_set2 = event_info[mid2]

                # 过滤 sentence 0 (URL/metadata)
                if sid1 == 0 or sid2 == 0:
                    self.skipped_sent0 += 1
                    continue

                # 确定 label：只有具备严格 Cause -> Effect 方向的才是正样本
                label = 0
                if (mid1, mid2) in causal_set:
                    label = 1
                    gt_causal.append((mid1, mid2))

                # 编码
                same_sent = (sid1 == sid2)
                if same_sent:
                    sent_text, f1, f2, e1_first = build_marked_sent_dual(
                        sid1, pos_set1, pos_set2)
                    if sent_text is None:
                        continue
                    if f1 and f2: self.span_hit += 1
                    else: self.span_miss += 1
                    pf = self._encode_pair(
                        esc_doc.doc_id, mid1, mid2, label,
                        sent_text, None, same_sent=True, e1_first=e1_first)
                else:
                    s1_text, f1 = build_marked_sent_single(sid1, pos_set1)
                    s2_text, f2 = build_marked_sent_single(sid2, pos_set2)
                    if s1_text is None or s2_text is None:
                        continue
                    if f1 and f2: self.span_hit += 1
                    else: self.span_miss += 1
                    pf = self._encode_pair(
                        esc_doc.doc_id, mid1, mid2, label,
                        s1_text, s2_text, same_sent=False, e1_first=True)

                if pf:
                    pair_features.append(pf)
                    self.total_pairs += 1
                    if label == 1: self.total_pos += 1
                    else: self.total_neg += 1

        if not pair_features:
            return None

        return DocFeature(
            doc_id=esc_doc.doc_id,
            topic_id=esc_doc.topic_id,
            event_keys=event_mids,
            pair_features=pair_features,
            gt_causal_pairs=gt_causal,
        )

    # ── RoBERTa 编码 ──────────────────────────────────────────
    def _encode_pair(self, doc_id, k1, k2, label,
                     sent1_text, sent2_text,
                     same_sent, e1_first=True) -> Optional[PairFeature]:
        if same_sent:
            enc = self.tok(sent1_text,
                           add_special_tokens=True,
                           max_length=self.max_len,
                           truncation=True)
        else:
            enc = self.tok(sent1_text, sent2_text,
                           add_special_tokens=True,
                           max_length=self.max_len,
                           truncation=True)

        ids = enc['input_ids']
        spans = self._find_spans(ids)

        if same_sent and len(spans) >= 2:
            if e1_first:
                e1s, e1e = spans[0]; e2s, e2e = spans[1]
            else:
                e1s, e1e = spans[1]; e2s, e2e = spans[0]
        elif len(spans) >= 2:
            e1s, e1e = spans[0]; e2s, e2e = spans[1]
        elif len(spans) == 1:
            e1s, e1e = spans[0]; e2s, e2e = spans[0]
        else:
            e1s, e1e = 1, min(3, len(ids) - 1)
            e2s, e2e = e1s, e1e

        return PairFeature(doc_id, k1, k2, ids,
                           enc['attention_mask'],
                           e1s, e1e, e2s, e2e, label)

    def _find_spans(self, ids: List[int]) -> List[Tuple[int, int]]:
        spans, start = [], None
        for i, t in enumerate(ids):
            if t == self.t_open:
                start = i + 1
            elif t == self.t_close and start is not None:
                spans.append((start, i))
                start = None
        return spans


# ── 独立测试 ─────────────────────────────────────────────────
if __name__ == '__main__':
    """
    用法: python processor_xml.py --esc_root ./v0.9 --model_name_or_path roberta-base
    打印统计信息，验证 XML 解析是否正确
    """
    import argparse
    from transformers import RobertaTokenizerFast

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--esc_root', default='./v0.9')
    parser.add_argument('--model_name_or_path', default='roberta-base')
    parser.add_argument('--max_seq_len', default=256, type=int)
    args = parser.parse_args()

    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name_or_path)
    tokenizer.add_tokens(['<t>', '</t>'])

    processor = XMLProcessor(args, tokenizer)
    docs = processor.build_all_docs(args.esc_root)

    # 统计
    topics = set()
    total_events = 0
    total_pairs = 0
    total_pos = 0
    total_intra = 0
    total_cross = 0
    intra_pos = 0
    cross_pos = 0

    for doc in docs:
        topics.add(doc.topic_id)
        total_events += len(doc.event_keys)
        for pf in doc.pair_features:
            total_pairs += 1
            if pf.label == 1:
                total_pos += 1
            # 判断 intra/cross: 如果 e1 和 e2 span 位置接近 → intra
            # 简单方法: 同句编码没有 </s></s> 分隔符

    logger.info(f"\n{'='*60}")
    logger.info(f"ESC v0.9 统计:")
    logger.info(f"  Topics: {len(topics)} → {sorted(topics)}")
    logger.info(f"  Documents: {len(docs)}")
    logger.info(f"  Total events: {total_events}")
    logger.info(f"  Total pairs: {total_pairs}")
    logger.info(f"  Positive (causal): {total_pos}")
    logger.info(f"  Negative: {total_pairs - total_pos}")
    logger.info(f"  Pos ratio: {total_pos/max(total_pairs,1)*100:.1f}%")

    # 原始 PLOT_LINK 统计
    raw_links = 0
    raw_skipped = 0
    raw_unique_directed = set()
    for doc_feat in docs:
        # 重新解析获取原始统计
        pass

    # 直接从 XML 重新扫一遍统计
    from lxml import etree as _et
    total_plot_links = 0
    total_plot_links_with_events = 0
    all_directed_pairs = set()
    for topic_id in sorted([d for d in os.listdir(args.esc_root)
                            if os.path.isdir(os.path.join(args.esc_root, d))]):
        topic_path = os.path.join(args.esc_root, topic_id)
        for xml_path in sorted(glob.glob(os.path.join(topic_path, '*ecbplus*'))):
            try:
                doc = ESCDocument(xml_path, topic_id)
                total_plot_links += doc._plot_link_total
                total_plot_links_with_events += (doc._plot_link_total - doc._plot_link_skipped)
                for s, t in doc.causal_pairs[::2]:  # 只取正向（每对存了双向）
                    all_directed_pairs.add((doc.doc_id, s, t))
            except:
                pass

    logger.info(f"\n  Raw PLOT_LINKs (with relType): {total_plot_links}")
    logger.info(f"  PLOT_LINKs where both src/tgt are events: {total_plot_links_with_events}")
    logger.info(f"  Unique directed (doc,src,tgt): {len(all_directed_pairs)}")
    logger.info(f"  Our undirected positive pairs: {total_pos}")
    logger.info(f"{'='*60}")

    # 对照论文数据: ESC 应有 22 topics, 258 docs, 5334 events, 5625 causal pairs
    # 注意: 5625 = 1770 intra + 3855 cross 是有向对的总数
    # 我们的 pair 生成是有向的，所以 positive 数应接近 5625 的去重值

    # 额外诊断: 统计原始 PLOT_LINK 数量
    raw_plot_links = 0
    raw_causal_links = 0   # 去重后有向对
    for doc in docs:
        # 重新解析统计
        pass

    logger.info(f"\n论文参考值:")
    logger.info(f"  Topics: 22 (got {len(topics)})")
    logger.info(f"  Docs: 258")
    logger.info(f"  Events: 5334")
    logger.info(f"  Causal pairs (directed): 5625 (1770 intra + 3855 cross)")
    logger.info(f"  我们的有向 causal pairs: {total_pos}")
    logger.info(f"  Total event pairs (directed): {total_pairs}")