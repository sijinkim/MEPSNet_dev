import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--cpu', action='store_true',
                        help='Using CPU only')
    parser.add_argument('--seed', type=int, default=0)

    # model
    parser.add_argument('--model', type=str, default='MEPSNet',
                        help='Select model')
    parser.add_argument('--pretrain', type=str,
                        help='path of the pretrained model')

    # dataset
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--dataset', type=str, default='DIV2K',
                        help='SHDD dataset(default: DIV2K)')
    parser.add_argument('--level', type=str, default='easy', choices=['easy', 'moderate', 'difficult'],
                        help='Level of dataset to validation and test (default:easy)')

    # training setups
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'l1'],
                        help='Loss type-mse or l1 (default:mse)')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='Weight decay value in Adam optimizer')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default:1e-3)')
    parser.add_argument('--decay', type=str,
                        default='200-500', help='Scheduler lr decay(default: at after 200th, 500th epochs [200-500])')
    parser.add_argument('--batchsize', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--patchsize', type=int, default=80,
                        help='Training data-crop size (default:80)')
    parser.add_argument('--extract_featsize', type=int, default=256,
                        help='Feature size in feature extraction module (default:256)')
    parser.add_argument('--num_experts', type=int, default=3)
    parser.add_argument('--expert_featsize', type=int, default=64,
                        help='Feature size in each expert module (default:64)')
    parser.add_argument('--eval_steps', type=int, default=int(50*9600),
                        help='Run validation every 50 epochs(1 epoch=9600 steps')
    parser.add_argument('--num_valimages', type=int, default=5,
                        help='Number of validation images (default:5)')
    parser.add_argument('--max_steps', type=int, default=2000*9600,
                        help='1 epoch has 9600 steps. Max epoch:2K(=2K*9600 steps')

    # misc
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--save_result', action='store_true',
                        help='Save restored im in evaluation phase')
    parser.add_argument('--ckpt_root', type=str, default='./pt')
    parser.add_argument('--save_root', type=str, default='./output')

    return parser.parse_args()


def make_template(opt):
    opt.strict_dict = opt.test_only

    # model
    if "mepsnet" in opt.model.lower():
        # optimal settings for MEPSNet
        opt.kernelsize = 3
        opt.num_templates = 16
        opt.num_SRIRs = 3
        opt.num_SResidualBlocks = 12


def get_option():
    opt = parse_args()
    make_template(opt)
    return opt
