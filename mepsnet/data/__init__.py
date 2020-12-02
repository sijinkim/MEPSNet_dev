import torch

from .dataset import SHDD_train, SHDD_test


def generate_loader(phase, opt):
    if phase == "train":
        dataset = SHDD_train(dataset_root=opt.dataset_root,
                             dataset=opt.dataset, size=opt.patchsize, level=opt.level)
    elif phase == "valid" or phase == "test":
        dataset = SHDD_test(
            dataset_root=opt.dataset_root, dataset=opt.dataset, level=opt.level, num_images=opt.num_valimages, type_=phase)
    else:
        raise ValueError(f"Unsupported dataset phase: {phase}")

    kwargs = {
        "batch_size": opt.batchsize if phase == "train" else 1,
        "num_workers": opt.num_workers if phase == "train" else 0,
        "shuffle": phase == "train",
        "drop_last": phase == "train"
    }

    return torch.utils.data.DataLoader(dataset, **kwargs)
