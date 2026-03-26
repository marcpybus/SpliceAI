import argparse
import logging
import pysam
from spliceai.utils import Annotator, get_delta_scores, INFO_FIELD_KEYS
import tensorflow as tf

try:
    from sys.stdin import buffer as std_in
    from sys.stdout import buffer as std_out
except ImportError:
    from sys import stdin as std_in
    from sys import stdout as std_out


def get_options():

    parser = argparse.ArgumentParser(description='Version: 1.3.1')
    parser.add_argument('-I', metavar='input', nargs='?', default=std_in,
                        help='path to the input VCF file, defaults to standard in')
    parser.add_argument('-O', metavar='output', nargs='?', default=std_out,
                        help='path to the output VCF file, defaults to standard out')
    parser.add_argument('-R', metavar='reference', required=True,
                        help='path to the reference genome fasta file')
    parser.add_argument('-A', metavar='annotation', required=True,
                        help='"grch37" (GENCODE V24lift37 canonical annotation file in '
                             'package), "grch38" (GENCODE V24 canonical annotation file in '
                             'package), or path to a similar custom gene annotation file')
    parser.add_argument('-D', metavar='distance', nargs='?', default=50,
                        type=int, choices=range(0, 5000),
                        help='maximum distance between the variant and gained/lost splice '
                             'site, defaults to 50')
    parser.add_argument('-M', metavar='mask', nargs='?', default=0,
                        type=int, choices=[0, 1],
                        help='mask scores representing annotated acceptor/donor gain and '
                             'unannotated acceptor/donor loss, defaults to 0')
    parser.add_argument('-C', metavar='c', nargs='?',
                        type=int, default=None,
                        help='Number of CPUs to use for TensorFlow threading (default: all)')
   
    args = parser.parse_args()

    return args


def main():

    args = get_options()

    if None in [args.I, args.O, args.D, args.M]:
        logging.error('Usage: spliceai [-h] [-I [input]] [-O [output]] -R reference -A annotation '
                      '[-D [distance]] [-M [mask]] [-C [cpus]]')
        exit()

    try:
        vcf = pysam.VariantFile(args.I)
    except (IOError, ValueError) as e:
        logging.error('{}'.format(e))
        exit()

    if args.C is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.C)
        tf.config.threading.set_inter_op_parallelism_threads(args.C)

    header = vcf.header
    header.add_line('##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAIv1.3.1 variant '
                    'annotation. These include delta scores (DS) and delta positions (DP) for '
                    'acceptor gain (AG), acceptor loss (AL), donor gain (DG), donor loss (DL), '
                    ' acceptor gain reference score (DS_AG_REF), acceptor gain variant score (DS_AG_ALT), '
                    ' acceptor loss reference score (DS_AL_REF), acceptor loss variant score (DS_AL_ALT), '
                    ' donor gain reference score (DS_DG_REF), donor gain variant score (DS_DG_ALT), '
                    ' donor loss reference score (DS_DL_REF), donor loss variant score (DS_DL_ALT).'
                    f'Format: {"|".join(INFO_FIELD_KEYS)}">')

    try:
        output = pysam.VariantFile(args.O, mode='w', header=header)
    except (IOError, ValueError) as e:
        logging.error('{}'.format(e))
        exit()

    ann = Annotator(args.R, args.A)

    for record in vcf:
        scores = get_delta_scores(record, ann, args.D, args.M)
        if len(scores) > 0:
            record.info['SpliceAI'] = [
                '|'.join([f"{d[key]:0.2f}" if isinstance(d[key], float) else str(d[key]) for key in INFO_FIELD_KEYS]) for d in scores
            ]
        output.write(record)

    vcf.close()
    output.close()


if __name__ == '__main__':
    main()
