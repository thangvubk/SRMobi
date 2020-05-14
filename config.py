import argparse

parser = argparse.ArgumentParser(description='PIRM 2018')

# dataset
parser.add_argument('--scale', type=int, default=4, help='interpolation scale. Default 4')
parser.add_argument('--train_dataset', type=str, default='DIV2K', help='Training dataset')
parser.add_argument('--valid_dataset', type=str, default='PIRM', help='Training dataset')
parser.add_argument('--num_valids', type=int, default=10, help='Number of image for validation')
# model
parser.add_argument('--num_channels', type=int, default=256, help='number of resnet channel')
parser.add_argument('--num_blocks', type=int, default=32, help='number of resnet blocks')
parser.add_argument('--res_scale', type=float, default=0.1)
parser.add_argument('--phase', type=str, default='train', help='phase: pretrain or train')
parser.add_argument(
    '--pretrained_model', type=str, default='', help='pretrained model for train phase (optional)')

# training
parser.add_argument('--batch_size', type=int, default=16, help='batch size used for training')
parser.add_argument(
    '--learning_rate',
    type=float,
    default=5e-5,
    help='learning rate used for training (use 1e-4 for pretrain)')
parser.add_argument('--lr_step', type=int, default=120, help='steps to decay learning rate')
parser.add_argument('--num_epochs', type=int, default=200, help='number of training epochs')
parser.add_argument(
    '--num_repeats', type=int, default=20, help='number of repeat per image for each epoch')
parser.add_argument('--patch_size', type=int, default=24, help='input patch size')


def get_args():
    args = parser.parse_args()
    return args
