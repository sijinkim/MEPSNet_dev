import importlib
import json
import numpy as np

import torch

from option import get_option
from solver import Solver


def main():
    opt = get_option()
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    module = importlib.import_module("model.{}".format(opt.model.lower()))

    if not opt.test_only:
        print(json.dumps(vars(opt), indent=4))

    solver = Solver(module, opt)
    if opt.test_only:
        print("Evaluate {} (loaded from {})".format(opt.model, opt.pretrain))
        psnr = solver.evaluate(solver.test_loader)
        print("{:.2f}".format(psnr))
    else:
        solver.fit()


if __name__ == "__main__":
    main()
