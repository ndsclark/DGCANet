import argparse
import torch
import models
from ptflops import get_model_complexity_info


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Parameters Test')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: emsanet50)')
parser.add_argument('--use_gpu', action='store_true', default=False,
                    help='Use NVIDIA GPU acceleration')


def main():
    args = parser.parse_args()

    # create model
    print("===> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    print("===> Successfully creating model '{}'".format(args.arch))

    # print Model architecture
    print("===> Print the constructing Model architecture '{}'".format(args.arch))
    print(model)

    # get the number of Model parameters
    print('Number of Model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    print("-" * 150)

    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    if args.use_gpu:
        model = model.cuda()
        print('===> Using GPU for acceleration')
    else:
        print('===> Using CPU for computation')

    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                              print_per_layer_stat=True, verbose=True, output_precision=3)
    print('{:<30}  {:<8}'.format('Computational complexity FLOPs: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == '__main__':
    main()
