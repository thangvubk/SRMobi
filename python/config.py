import argparse

parser = argparse.ArgumentParser(description='MobiSR')

# dataset
parser.add_argument('--train_dataset', type=str, default='DIV2K', help='Training dataset')
parser.add_argument('--valid_dataset', type=str, default='DIV2K', help='Validation dataset')
parser.add_argument('--test_dataset', type=str, default='DIV2K', help='Test datset')
parser.add_argument('--num_valids', type=int, default=10, help='Number of image for validation')
parser.add_argument('--patch_size', type=int, default=48, help='Input patch size of train dataset')

# model
parser.add_argument('--num_blocks', type=int, default=16, help='number of resnet block')
parser.add_argument('--num_channels', type=int, default=16, help='number of resnet channel')

# training
parser.add_argument('--batch_size', type=int, default=16, help='batch size used for training')
parser.add_argument(
    '--learning_rate', type=float, default=1e-4, help='inital training learning rate')
parser.add_argument('--lr_step', type=int, default=120, help='steps to decay learning rate')
parser.add_argument('--num_epochs', type=int, default=200, help='number of training epochs')

# testing
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--result_dir', type=str, default='./results')
parser.add_argument('--write_results', action='store_true', help='weather write result images')

# misc
parser.add_argument('--eval_freq', type=int, default=10, help='evaluation frequency (epochs)')
parser.add_argument('--work_dir', type=str, default='./work_dirs/edsr')


def get_args():
    args = parser.parse_args()
    return args
