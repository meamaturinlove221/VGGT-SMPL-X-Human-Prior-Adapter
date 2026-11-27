"""本轮真实B绑定后的固定训练与评价入口。"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.dna_b_adapter_20261021.paired_train import train
from research.dna_b_adapter_20261021.evaluate import evaluate


def main():
    """只接受已保存的真实合同与清单，不创建新模型工厂。"""
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='action',required=True)
    fit=sub.add_parser('train');fit.add_argument('--contract',required=True)
    ev=sub.add_parser('evaluate');ev.add_argument('--checkpoint',required=True)
    ev.add_argument('--manifest',required=True);ev.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.action=='train':train(args.contract)
    else:evaluate(args.checkpoint,args.manifest,args.output)


if __name__=='__main__':
    main()
