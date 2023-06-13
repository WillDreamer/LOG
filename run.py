import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import random
from loguru import logger
import method
from model.model_loader import load_model
import warnings
warnings.filterwarnings("ignore")

def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

multi_labels_dataset = [
    'nus-wide-tc-10',
    'nus-wide-tc-21',
    'flickr25k',
    'coco'
]

num_features = {
    'alexnet': 4096,
    'vgg16': 4096,
}


def run():
    # Load configuration
    seed_torch()
    seed = 2022
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    args = load_config()
    logger.add(os.path.join('logs', '{time}.log'), rotation="500 MB", level="INFO")
    logger.info(args.setting)
    logger.info(args.tag)
    logger.info(args.source.split('/')[-1])
    logger.info(args.target.split('/')[-1])

    if args.tag == 'officehome':
        from data.officehome import load_data
    elif args.tag == 'office':
        from data.office31 import load_data
    else:
        from data.visda import load_data
    


    # Load dataset
    query_dataloader, train_s_dataloader, retrieval_dataloader \
        = load_data(args.source, args.target,args.batch_size,args.num_workers,task=args.setting)
        
    if args.train:
        method.train(
            train_s_dataloader,
            query_dataloader,
            retrieval_dataloader,
            args.code_length,
            args.max_iter,
            args.arch,
            args.lr,
            args.device,
            args.verbose,
            args.topk,
            args.num_class,
            args.evaluate_interval,
            args.tag,
            args.batch_size,
            args.knn,
        )
    elif args.evaluate:
        model = load_model(args.arch, args.code_length)
        #model = nn.DataParallel(model,device_ids=[0,1,2])
        model_checkpoint = torch.load('./checkpoints/resume_64.t')
        model.load_state_dict(model_checkpoint['model_state_dict'])
        mAP = method.evaluate(
            model,
            query_dataloader,
            retrieval_dataloader,
            args.code_length,
            args.device,
            args.topk,
            )

    else:
        raise ValueError('Error configuration, please check your config, using "train", "resume" or "evaluate".')


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='DG_PyTorch')
    
    
    parser.add_argument('--tag', type=str, default='officehome', help="Tag")
    parser.add_argument('--root', type=str, default='/data/hash_data/office-home/', help="root")
    parser.add_argument('--source', type=str, default='Product.txt', help="The source dataset")
    parser.add_argument('--target', type=str, default='Real_World.txt', help="The target dataset")
    parser.add_argument('--num_class', default=65, type=int,
                        help='Source Class')
    
    
    parser.add_argument('--setting', type=str, default='cross', help="cross or single")
    parser.add_argument('-k', '--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: -1)')
    parser.add_argument('-c', '--code-length', default=64, type=int,
                        help='Binary hash code length.(default: 64)')
    parser.add_argument('-l', '--lr', default=1e-3, type=float,
                        help='Learning rate.(default: 1e-3)')
    parser.add_argument('-w', '--num-workers', default=0, type=int,
                        help='Number of loading data threads.(default: 0)')
    parser.add_argument('-b', '--batch-size', default=24, type=int,
                        help='Batch size.(default: 24)')
    parser.add_argument('-a', '--arch', default='vgg16', type=str,
                        help='CNN architecture.(default: vgg16)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print log.')
    parser.add_argument('--train', action='store_true',
                        help='Training mode.')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluation mode.')
    parser.add_argument('-g', '--gpu', default=0, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('-n', '--knn', default=20, type=int,
                        help='Knn.(default: 10)')
    parser.add_argument('--temperature', default=0.5, type=float,
                        help='Hyper-parameter in SimCLR .(default:0.5)')
    parser.add_argument('-T', '--max-iter', default=50, type=int,
                        help='Number of iterations.(default: 150)')
    parser.add_argument('-e', '--evaluate-interval', default=10, type=int,
                        help='Interval of evaluation.(default: 500)')
    
    

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)
        torch.cuda.set_device(args.gpu)
    
    args.source = os.path.join(args.root,args.source)
    args.target = os.path.join(args.root,args.target)
    return args


if __name__ == '__main__':
    run()
