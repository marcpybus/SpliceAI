from pkg_resources import resource_filename

import pandas as pd
import numpy as np
from pyfastx import Fasta
import onnxruntime as ort
import logging
import os
from sys import exit
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------------------
# ONNX Runtime configuration for AMD EPYC
# Adjust PHYSICAL_CORES to match your CPU (e.g. 64 for EPYC 7763)
# ---------------------------------------------------------------------------
PHYSICAL_CORES = int(os.environ.get('OMP_NUM_THREADS', 64))

def _make_ort_session(model_path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = PHYSICAL_CORES
    opts.inter_op_num_threads = 4
    opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    opts.enable_cpu_mem_arena = True
    opts.enable_mem_pattern = True
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=['CPUExecutionProvider']
    )

# ---------------------------------------------------------------------------
# One-time ONNX conversion helper
# Run convert_models_to_onnx() once before using the Annotator.
# It will produce spliceai1_ort.onnx ... spliceai5_ort.onnx next to the .h5 files.
# ---------------------------------------------------------------------------
def convert_models_to_onnx(output_dir: str = None):
    """
    Convert the 5 SpliceAI Keras .h5 models to ONNX format.
    Requires tensorflow and tf2onnx to be installed (only needed once).
    After conversion you can uninstall tensorflow if desired.

    Args:
        output_dir: directory to write .onnx files.
                    Defaults to the spliceai package models/ directory.
    """
    try:
        import tensorflow as tf
        import tf2onnx
    except ImportError:
        raise ImportError(
            "tensorflow and tf2onnx are required for conversion.\n"
            "Install with:  pip install tensorflow tf2onnx\n"
            "You can uninstall them after conversion."
        )

    for i in range(1, 6):
        h5_path = resource_filename('spliceai', f'models/spliceai{i}.h5')
        out_path = os.path.join(
            output_dir or os.path.dirname(h5_path),
            f'spliceai{i}_ort.onnx'
        )

        if os.path.exists(out_path):
            logging.info(f'ONNX model already exists, skipping: {out_path}')
            continue

        logging.info(f'Converting {h5_path} -> {out_path} ...')
        model = tf.keras.models.load_model(h5_path)

        # SpliceAI input: (batch, 10000 + 2*dist_var, 4)
        # Use None for batch and sequence length so the model accepts any size
        input_signature = [
            tf.TensorSpec(shape=(None, None, 4), dtype=tf.float32, name='input')
        ]

        tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=17,
            output_path=out_path
        )
        logging.info(f'  Saved: {out_path}')

    logging.info('All models converted.')


def _onnx_model_paths() -> list:
    """Return the 5 .onnx paths, raising a clear error if they don't exist."""
    paths = []
    for i in range(1, 6):
        p = resource_filename('spliceai', f'models/spliceai{i}_ort.onnx')
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"ONNX model not found: {p}\n"
                "Run convert_models_to_onnx() once to generate the .onnx files."
            )
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Constants (unchanged from original)
# ---------------------------------------------------------------------------
INFO_FIELD_KEYS = [
    'ALLELE',
    'NAME',
    'STRAND',
    'DS_AG',
    'DS_AL',
    'DS_DG',
    'DS_DL',
    'DP_AG',
    'DP_AL',
    'DP_DG',
    'DP_DL',
    'DS_AG_REF',
    'DS_AG_ALT',
    'DS_AL_REF',
    'DS_AL_ALT',
    'DS_DG_REF',
    'DS_DG_ALT',
    'DS_DL_REF',
    'DS_DL_ALT',
]

FLOAT_FORMAT = "0.2f"
MIN_SCORE_THRESHOLD = 0.01
INSERTED_BASES_CONTEXT = 5


# ---------------------------------------------------------------------------
# Annotator — only __init__ changed: loads ORT sessions instead of Keras models
# ---------------------------------------------------------------------------
class Annotator:

    def __init__(self, ref_fasta, annotations):

        if annotations == 'grch37':
            annotations = resource_filename(__name__, 'annotations/grch37.txt')
        elif annotations == 'grch38':
            annotations = resource_filename(__name__, 'annotations/grch38.txt')

        try:
            df = pd.read_csv(annotations, sep='\t', dtype={'CHROM': object})
            self.genes = df['#NAME'].to_numpy()
            self.chroms = df['CHROM'].to_numpy()
            self.strands = df['STRAND'].to_numpy()
            self.tx_starts = df['TX_START'].to_numpy()+1
            self.tx_ends = df['TX_END'].to_numpy()
            self.exon_starts = [np.asarray([int(i) for i in c.split(',') if i])+1
                                 for c in df['EXON_START'].to_numpy()]
            self.exon_ends = [np.asarray([int(i) for i in c.split(',') if i])
                               for c in df['EXON_END'].to_numpy()]
        except IOError as e:
            logging.error('{}'.format(e))
            exit()
        except (KeyError, pd.errors.ParserError) as e:
            logging.error('Gene annotation file {} not formatted properly: {}'.format(annotations, e))
            exit()

        try:
            self.ref_fasta = Fasta(ref_fasta)
        except IOError as e:
            logging.error('{}'.format(e))
            exit()

        # --- CHANGED: load ONNX Runtime sessions instead of Keras models ---
        logging.info('Loading ONNX Runtime sessions (AMD EPYC optimized)...')
        self.models = [_make_ort_session(p) for p in _onnx_model_paths()]
        # Cache the input name once (same for all 5 models)
        self._input_name = self.models[0].get_inputs()[0].name
        logging.info(f'Loaded {len(self.models)} ORT sessions '
                     f'(intra_op_threads={PHYSICAL_CORES})')
        # --------------------------------------------------------------------

    def get_name_and_strand(self, chrom, pos):
        chrom = normalise_chrom(chrom, list(self.chroms)[0])
        idxs = np.intersect1d(np.nonzero(self.chroms == chrom)[0],
                               np.intersect1d(np.nonzero(self.tx_starts <= pos)[0],
                                              np.nonzero(pos <= self.tx_ends)[0]))
        if len(idxs) >= 1:
            return self.genes[idxs], self.strands[idxs], idxs
        else:
            return [], [], []

    def get_pos_data(self, idx, pos):
        dist_tx_start = self.tx_starts[idx]-pos
        dist_tx_end = self.tx_ends[idx]-pos
        dist_exon_bdry = min(np.union1d(self.exon_starts[idx], self.exon_ends[idx])-pos, key=abs)
        dist_ann = (dist_tx_start, dist_tx_end, dist_exon_bdry)
        return dist_ann


# ---------------------------------------------------------------------------
# Helpers (unchanged)
# ---------------------------------------------------------------------------
def one_hot_encode(seq):
    map = np.asarray([[0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')
    return map[np.fromstring(seq, np.int8) % 5]


def normalise_chrom(source, target):
    def has_prefix(x):
        return x.startswith('chr')
    if has_prefix(source) and not has_prefix(target):
        return source.strip('chr')
    elif not has_prefix(source) and has_prefix(target):
        return 'chr'+source
    return source


# ---------------------------------------------------------------------------
# get_delta_scores_for_transcript
# Only the model.predict() calls are changed — everything else is identical.
# ---------------------------------------------------------------------------
def get_delta_scores_for_transcript(x_ref, x_alt, ref_len, alt_len, strand, cov, ann):
    del_len = max(ref_len-alt_len, 0)

    x_ref = one_hot_encode(x_ref)[None, :].astype(np.float32)   # (1, L, 4)
    x_alt = one_hot_encode(x_alt)[None, :].astype(np.float32)

    if strand == '-':
        x_ref = x_ref[:, ::-1, ::-1]
        x_alt = x_alt[:, ::-1, ::-1]

    # --- CHANGED: ORT inference replaces model.predict() ---
    # Run all 5 sessions in parallel (mirrors the original ThreadPoolExecutor pattern)
    input_name = ann._input_name

    def _run(sess_x):
        sess, x = sess_x
        return sess.run(None, {input_name: x})[0]   # shape: (1, L, 3)

    with ThreadPoolExecutor(max_workers=5) as executor:
        preds_ref = list(executor.map(_run, [(sess, x_ref) for sess in ann.models]))
    y_ref = np.mean(preds_ref, axis=0)

    with ThreadPoolExecutor(max_workers=5) as executor:
        preds_alt = list(executor.map(_run, [(sess, x_alt) for sess in ann.models]))
    y_alt = np.mean(preds_alt, axis=0)
    # --------------------------------------------------------

    if strand == '-':
        y_ref = y_ref[:, ::-1]
        y_alt = y_alt[:, ::-1]

    y_alt_with_inserted_bases = None

    if ref_len > 1 and alt_len == 1:
        y_alt = np.concatenate([
            y_alt[:, :cov//2+alt_len],
            np.zeros((1, del_len, 3)),
            y_alt[:, cov//2+alt_len:]],
            axis=1)

    elif ref_len == 1 and alt_len > 1:
        y_alt_with_inserted_bases = y_alt
        y_alt = np.concatenate([
            y_alt[:, :cov//2],
            np.max(y_alt[:, cov//2:cov//2+alt_len], axis=1)[:, None, :],
            y_alt[:, cov//2+alt_len:]],
            axis=1)

    # MNP handling
    elif ref_len > 1 and alt_len > 1:
        zblock = np.zeros((1, ref_len-1, 3))
        y_alt = np.concatenate([
            y_alt[:, :cov//2],
            np.max(y_alt[:, cov//2:cov//2+alt_len], axis=1)[:, None, :],
            zblock,
            y_alt[:, cov//2+alt_len:]],
            axis=1)

    return y_ref, y_alt, y_alt_with_inserted_bases


# ---------------------------------------------------------------------------
# get_delta_scores — completely unchanged
# ---------------------------------------------------------------------------
def get_delta_scores(record, ann, dist_var, mask):
    cov = 2*dist_var+1
    wid = 10000+cov
    scores = []

    try:
        record.chrom, record.pos, record.ref, len(record.alts)
    except TypeError:
        logging.warning('Skipping record (bad input): {}'.format(record))
        return scores

    (genes, strands, idxs) = ann.get_name_and_strand(record.chrom, record.pos)
    if len(idxs) == 0:
        return scores

    chrom = normalise_chrom(record.chrom, list(ann.ref_fasta.keys())[0])

    try:
        seq = ann.ref_fasta[chrom][record.pos-wid//2-1:record.pos+wid//2].seq
    except (IndexError, ValueError):
        logging.warning('Skipping record (fasta issue): {}'.format(record))
        return scores

    if seq[wid//2:wid//2+len(record.ref)].upper() != record.ref:
        logging.warning('Skipping record (ref issue): {}'.format(record))
        return scores

    if len(seq) != wid:
        logging.warning('Skipping record (near chromosome end): {}'.format(record))
        return scores

    if len(record.ref) > 2*dist_var:
        logging.warning('Skipping record (ref too long): {}'.format(record))
        return scores

    genomic_coords = np.arange(record.pos - cov//2, record.pos + cov//2 + 1)

    delta_scores_transcript_cache = {}

    for j in range(len(record.alts)):
        for i in range(len(idxs)):

            if '.' in record.alts[j] or '-' in record.alts[j] or '*' in record.alts[j]:
                continue
            if '<' in record.alts[j] or '>' in record.alts[j]:
                continue

            dist_ann = ann.get_pos_data(idxs[i], record.pos)
            pad_size = [max(wid//2+dist_ann[0], 0), max(wid//2-dist_ann[1], 0)]

            ref_len = len(record.ref)
            alt_len = len(record.alts[j])

            x_ref = 'N'*pad_size[0]+seq[pad_size[0]:wid-pad_size[1]]+'N'*pad_size[1]
            x_alt = x_ref[:wid//2]+str(record.alts[j])+x_ref[wid//2+ref_len:]

            strand = strands[i]
            args = (x_ref, x_alt, ref_len, alt_len, strand, cov)

            if args not in delta_scores_transcript_cache:
                delta_scores_transcript_cache[args] = get_delta_scores_for_transcript(*args, ann=ann)

            y_ref, y_alt, y_alt_with_inserted_bases = delta_scores_transcript_cache[args]

            y = np.concatenate([y_ref, y_alt])

            idx_pa = (y[1, :, 1]-y[0, :, 1]).argmax()
            idx_na = (y[0, :, 1]-y[1, :, 1]).argmax()
            idx_pd = (y[1, :, 2]-y[0, :, 2]).argmax()
            idx_nd = (y[0, :, 2]-y[1, :, 2]).argmax()

            mask_pa = np.logical_and((idx_pa-cov//2 == dist_ann[2]), mask)
            mask_na = np.logical_and((idx_na-cov//2 != dist_ann[2]), mask)
            mask_pd = np.logical_and((idx_pd-cov//2 == dist_ann[2]), mask)
            mask_nd = np.logical_and((idx_nd-cov//2 != dist_ann[2]), mask)

            if len(genomic_coords) != y_ref.shape[1]:
                raise ValueError(f"SpliceAI internal error: len(genomic_coords) != y_ref.shape[1]: "
                                  f"{len(genomic_coords)} != {y_ref.shape[1]}")

            if len(genomic_coords) != y_alt.shape[1]:
                raise ValueError(f"SpliceAI internal error: len(genomic_coords) != y_alt.shape[1]: "
                                  f"{len(genomic_coords)} != {y_alt.shape[1]}")

            DS_AG = (y[1, idx_pa, 1]-y[0, idx_pa, 1])*(1-mask_pa)
            DS_AL = (y[0, idx_na, 1]-y[1, idx_na, 1])*(1-mask_na)
            DS_DG = (y[1, idx_pd, 2]-y[0, idx_pd, 2])*(1-mask_pd)
            DS_DL = (y[0, idx_nd, 2]-y[1, idx_nd, 2])*(1-mask_nd)

            DP_AG = int(idx_pa-cov//2)
            DP_AL = int(idx_na-cov//2)
            DP_DG = int(idx_pd-cov//2)
            DP_DL = int(idx_nd-cov//2)

            if ref_len == 1 and alt_len > 1 and ((DS_AG >= 0.01 and DP_AG == 0) or (DS_DG >= 0.01 and DP_DG == 0)):
                inserted_bases_genomic_coords = np.concatenate([
                    np.arange(record.pos - INSERTED_BASES_CONTEXT + 1, record.pos + 1),
                    [f"+{offset}" for offset in np.arange(1, alt_len)],
                    np.arange(record.pos + 1, record.pos + INSERTED_BASES_CONTEXT + 1),
                ])

                y_ref_inserted_bases = np.concatenate([
                    y_ref[:, 1 + cov//2 - INSERTED_BASES_CONTEXT : 1 + cov//2],
                    np.zeros((1, alt_len - 1, 3)),
                    y_ref[:, 1 + cov//2 : 1 + cov//2 + INSERTED_BASES_CONTEXT],
                ], axis=1)

                y_alt_inserted_bases = y_alt_with_inserted_bases[
                    :, 1 + cov//2 - INSERTED_BASES_CONTEXT: 1 + cov//2 + (alt_len - 1) + INSERTED_BASES_CONTEXT]

                assert y_ref_inserted_bases.shape == y_alt_inserted_bases.shape

                ref_seq = (
                    seq[wid//2 - INSERTED_BASES_CONTEXT + 1: wid//2 + 1] +
                    " " * (alt_len - 1) +
                    seq[wid//2 + 1 : wid//2 + 1 + INSERTED_BASES_CONTEXT]
                )
                alt_seq = (
                    seq[wid//2 - INSERTED_BASES_CONTEXT + 1: wid//2 + 1] +
                    record.alts[j][1:] +
                    seq[wid//2 + len(record.ref) : wid//2 + len(record.ref) + INSERTED_BASES_CONTEXT]
                )
                assert len(ref_seq) == len(alt_seq), f"len(ref_seq) != len(alt_seq): {len(ref_seq)} != {len(alt_seq)}"

            else:
                inserted_bases_genomic_coords = ref_seq = alt_seq = y_ref_inserted_bases = y_alt_inserted_bases = None

            scores.append({
                "ALLELE": record.alts[j],
                "NAME": genes[i],
                "STRAND": strands[i],
                "DS_AG": f"{DS_AG:{FLOAT_FORMAT}}",
                "DS_AL": f"{DS_AL:{FLOAT_FORMAT}}",
                "DS_DG": f"{DS_DG:{FLOAT_FORMAT}}",
                "DS_DL": f"{DS_DL:{FLOAT_FORMAT}}",
                "DP_AG": DP_AG,
                "DP_AL": DP_AL,
                "DP_DG": DP_DG,
                "DP_DL": DP_DL,
                "DS_AG_REF": f"{y[0, idx_pa, 1]:{FLOAT_FORMAT}}",
                "DS_AL_REF": f"{y[0, idx_na, 1]:{FLOAT_FORMAT}}",
                "DS_DG_REF": f"{y[0, idx_pd, 2]:{FLOAT_FORMAT}}",
                "DS_DL_REF": f"{y[0, idx_nd, 2]:{FLOAT_FORMAT}}",
                "DS_AG_ALT": f"{y[1, idx_pa, 1]:{FLOAT_FORMAT}}",
                "DS_AL_ALT": f"{y[1, idx_na, 1]:{FLOAT_FORMAT}}",
                "DS_DG_ALT": f"{y[1, idx_pd, 2]:{FLOAT_FORMAT}}",
                "DS_DL_ALT": f"{y[1, idx_nd, 2]:{FLOAT_FORMAT}}",
                "ALL_NON_ZERO_SCORES": [
                    {
                        "pos": int(genomic_coord),
                        "RA": f"{ref_acceptor_score:{FLOAT_FORMAT}}",
                        "AA": f"{alt_acceptor_score:{FLOAT_FORMAT}}",
                        "RD": f"{ref_donor_score:{FLOAT_FORMAT}}",
                        "AD": f"{alt_donor_score:{FLOAT_FORMAT}}",
                    } for i, (genomic_coord, ref_acceptor_score, alt_acceptor_score, ref_donor_score, alt_donor_score) in enumerate(zip(
                        genomic_coords, y_ref[0, :, 1], y_alt[0, :, 1], y_ref[0, :, 2], y_alt[0, :, 2])
                    ) if any(score >= MIN_SCORE_THRESHOLD for score in (ref_acceptor_score, alt_acceptor_score, ref_donor_score, ref_acceptor_score))
                    or i in (idx_pa, idx_na, idx_pd, idx_nd)
                ],
                "SCORES_FOR_INSERTED_BASES": [] if y_alt_inserted_bases is None else [
                    {
                        "chrom": chrom,
                        "pos": genomic_coord,
                        "ref": ref_base,
                        "alt": alt_base,
                        "RA": f"{ref_acceptor_score:{FLOAT_FORMAT}}",
                        "RD": f"{ref_donor_score:{FLOAT_FORMAT}}",
                        "AA": f"{alt_acceptor_score:{FLOAT_FORMAT}}",
                        "AD": f"{alt_donor_score:{FLOAT_FORMAT}}",
                    } for i, (genomic_coord, ref_base, alt_base, ref_acceptor_score, alt_acceptor_score, ref_donor_score, alt_donor_score) in enumerate(zip(
                        inserted_bases_genomic_coords, ref_seq, alt_seq,
                        y_ref_inserted_bases[0, :, 1], y_alt_inserted_bases[0, :, 1],
                        y_ref_inserted_bases[0, :, 2], y_alt_inserted_bases[0, :, 2]))
                ],
            })

    return scores