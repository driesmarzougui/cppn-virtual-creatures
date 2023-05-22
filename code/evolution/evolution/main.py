import argparse
import sys
from lle import main as lle
from single import main as single
from eval_model import main as eval
import ray

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EVO-NM")
    parser.add_argument("--mode", type=str, default="single")
    parser.add_argument("--cluster", type=bool, default=False)
    parser.add_argument("--params", type=str, default="meta/params/params2_big_8.json")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--eval-path", type=str, default=None)
    args = parser.parse_args()

    sys.setrecursionlimit(3200)

    if args.mode == "single":
        single(args)
    elif args.mode == "lle":
        lle(args)
    elif args.mode == "eval":
        assert args.eval_path is not None, "Eval mode requires the eval-path CLI argument"
        eval(args)
