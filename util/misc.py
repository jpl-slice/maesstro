# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import random
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

try:
    from torch._six import inf
except ImportError:
    from torch import inf


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    """
    Log metrics during the training of a machine learning model.

    Example usage (at every epoch):
        logger = misc.MetricLogger(delimiter="\t")
        iterable = logger.log_every(data_loader, print_freq, header="Epoch: [5/10]")
        for data_iter_step, (images, labels) in enumerate(iterable):
            loss, acc = train_model()
            logger.update(loss=loss, accuracy=acc)
        logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        stats = {metric: meter.global_avg for metric, meter in logger.meters.items()}
    """

    def __init__(self, delimiter="\t"):
        """
        Initializes a new MetricLogger object.

        Parameters:
            delimiter (str): The delimiter to use when printing the metrics. Default is "\t".
        """
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.meters["iter_time"] = SmoothedValue(fmt="{avg:.4f}")
        self.meters["data_time"] = SmoothedValue(fmt="{avg:.4f}")

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """This function iterates over an iterable object and yields the elements of the iterable.
        It also logs the metrics stored in the meters dictionary at regular intervals
        specified by the print_freq argument.
        It takes an optional `header` argument as a title for the log message.
        It also includes the elapsed time and memory usage in the log message if GPU is available.

        Args:
            iterable (DataLoader): PyTorch DataLoader that returns batches of training data
            print_freq (int): Print interval for self.meters dictionary
            header (str, optional): Title for the log message. Defaults to None.

        Yields:
            torch.Tensor: A batch of PyTorch tensors that represent train/val data
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = self.meters["iter_time"]  # time to iterate over a batch of data
        data_time = self.meters["data_time"]  # time taken to load batch of data
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f} MB")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj  # yield does not terminate execution of the function
            # re-enter after training loop; iter_time includes forward/back prop
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(gpu, args):
    # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]

    try:  # try submitit first
        import submitit

        job_env = submitit.JobEnvironment()
        args.local_rank = job_env.local_rank  # range: 0 to (num_gpus_per_node - 1)
        args.global_rank = job_env.global_rank  # 0 to (num_nodes * gpus_per_node - 1)
        args.world_size = job_env.num_tasks
        args.node_rank = job_env.node
        gpu = args.local_rank
    except (
        Exception
    ) as e:  # try to use SLURM vars if local_rank and node_rank not specified
        print(
            f"Encountered '{e}' when loading distributed vars "
            f"from submitit. Using environment vars instead."
        )
        if args.local_rank == -1:
            args.local_rank = int(os.environ.get("SLURM_LOCALID", gpu))
        if args.node_rank == -1:
            args.node_rank = int(
                os.environ.get("SLURM_NODEID", os.environ.get("RANK", 0))
            )
        # https://github.com/facebookincubator/submitit/blob/main/submitit/slurm/slurm.py#L179
        args.global_rank = int(
            os.environ.get(
                "SLURM_PROCID", args.node_rank * args.ngpus_per_node + args.local_rank
            )
        )
        args.world_size = args.nodes * args.ngpus_per_node
        if "SLURM_SUBMIT_HOST" in os.environ:
            args.master_addr = os.environ["SLURM_SUBMIT_HOST"]

    if args.world_size > 1 and not args.multiprocessing_distributed:
        warnings.warn(
            f"WARNING:\n"
            f"========\n"
            f"World size is {args.world_size}, but neither DDP nor DataParallel"
            f" are enabled. Setting gpus=1, nodes=1, dataparallel=False, ddp=False"
        )
        args.gpus = args.nodes = args.world_size = 1
        args.multiprocessing_distributed = False
    elif args.world_size == 1 and args.multiprocessing_distributed:
        warnings.warn(
            f"WARNING:\n"
            f"========\n"
            f"World size is 1, but DDP is enabled. Setting gpus=1, nodes=1,"
            f" dataparallel=False, ddp=False"
        )
        args.gpus = args.nodes = args.world_size = 1
        args.multiprocessing_distributed = False

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.distributed:
        if "dist_url" not in args:
            args.dist_url = f"tcp://{args.master_addr}:{args.port}"
        print(
            f"Initializing process group on global rank {args.global_rank} "
            f"on node {args.node_rank}, gpu {args.local_rank} "
            f"with port {args.port} on {args.dist_url}"
        )
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.global_rank,
        )
        print(f"Process initialization completed for global rank {args.global_rank}")

    elif gpu is None and not torch.cuda.is_available():
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    args.device = torch.device(
        f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    )
    torch.cuda.set_device(args.device)

    # suppress printing if not master
    if args.multiprocessing_distributed and args.global_rank != 0:
        # only print on master node
        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass


def set_manual_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        # set numpy seed as well
        np.random.seed(seed)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)
    return os.path.join(args.output_dir, f"checkpoint-{epoch_name}.pth")


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
