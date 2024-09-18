# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import os
import shutil
import subprocess

import submitit

from util.parse_args import parse_args


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import os
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        import main_pretrain as trainer

        dist_env = submitit.helpers.TorchDistributedEnvironment()  # .export()
        self.set_hostname_port()
        if dist_env.local_rank == 0:
            self.copy_files_to_tmp()
        else:
            # wait for copy files to finish
            import time

            time.sleep(1500)  # (~20 mins)
        trainer.main(self.args)

    def checkpoint(self):
        import os

        import submitit

        from util.misc import init_distributed_mode

        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def set_hostname_port(self):
        # find a common host name on all nodes
        cmd = "scontrol show hostnames " + os.getenv("SLURM_JOB_NODELIST")
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        self.args.master_addr = host_name
        self.args.dist_url = f"tcp://{host_name}:{self.args.port}"
        os.environ["MASTER_ADDR"] = host_name
        os.environ["MASTER_PORT"] = self.args.port

    def copy_files_to_tmp(self, parallel=True):
        """Copy training files to /tmp or /dev/shm to speed up I/O"""
        import os
        import shutil

        train_data = self.args.data_path
        val_data = vars(self.args).get("val_data_path", "")

        tmp = "/tmp"
        shm = "/dev/shm"

        tmp_space = int(0.85 * shutil.disk_usage(tmp).free)  # 288G
        shm_space = int(0.65 * shutil.disk_usage(shm).free)  # 126G

        train_data_size = get_size(train_data)
        val_data_size = get_size(val_data)

        self.copy_data(
            train_data, tmp, shm, tmp_space, shm_space, train_data_size, parallel
        )

        # get updated tmp_space and shm_space
        tmp_space = shutil.disk_usage(tmp).free
        shm_space = shutil.disk_usage(shm).free

        self.copy_data(
            val_data,
            tmp,
            shm,
            tmp_space,
            shm_space,
            val_data_size,
            is_train=False,
            parallel=parallel,
        )

    def copy_data(
        self,
        data,
        tmp,
        shm,
        tmp_space,
        shm_space,
        data_size,
        is_train=True,
        parallel=True,
    ):
        if os.path.isfile(data) and (data_size < tmp_space or data_size < shm_space):
            target = tmp if data_size < tmp_space else shm
            self.args.data_path = os.path.join(target, os.path.basename(data))
            if not os.path.exists(self.args.data_path):
                print(
                    f"Copying {data} (size: {data_size/1e9:.2f} GB) to {self.args.data_path}"
                )
                shutil.copy(data, self.args.data_path)
        elif os.path.isdir(data):  # and data_size < shm_space + tmp_space:
            # split the data between /tmp and /dev/shm, and create a symlink to the data in /tmp
            os.makedirs(os.path.join(tmp, os.path.basename(data)), exist_ok=True)
            os.makedirs(os.path.join(shm, os.path.basename(data)), exist_ok=True)
            # step 2: make a list of all files, their paths, and their sizes
            files_and_sizes = dict()
            for dirpath, dirnames, filenames in os.walk(data):
                for f in filenames:
                    filepath = os.path.join(dirpath, f)
                    files_and_sizes[filepath] = os.path.getsize(filepath)
            # step 3: sort the dictionary by size; we want to copy the largest files first
            files_and_sizes = {
                k: v
                for k, v in sorted(
                    files_and_sizes.items(), key=lambda item: item[1], reverse=True
                )
            }

            # step 3.5: print current diagnostics
            print(
                f"Found {len(files_and_sizes.keys())} files in {data},"
                f" totalling {sum(s for s in files_and_sizes.values())/1e9:.2f} GB."
                f"\nCurrent available space in /tmp: {tmp_space/1e9:.2f} GB."
                f"\nCurrent available space in /dev/shm: {shm_space/1e9:.2f} GB."
            )

            # step 4: copy files to tmp until tmp is full
            tmp_space_left, shm_space_left = tmp_space, shm_space
            final_filepaths = []
            from tqdm.auto import tqdm

            if not parallel:
                for filepath, size in tqdm(files_and_sizes.items()):
                    if size < tmp_space_left:  # copy file to tmp
                        final_filepaths.append(copy_file(filepath, data, tmp))
                        tmp_space_left -= size
                    elif size < shm_space_left:  # copy file to shm
                        final_filepaths.append(copy_file(filepath, data, shm))
                        shm_space_left -= size
                    else:  # no space left
                        final_filepaths.append(filepath)
            else:
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=64
                ) as executor, tqdm(total=len(files_and_sizes.items())) as progress_bar:
                    futures = []
                    for filepath, size in files_and_sizes.items():
                        if size < shm_space_left:
                            futures.append(
                                executor.submit(copy_file, filepath, data, shm)
                            )
                            shm_space_left -= size
                        elif size < tmp_space_left:
                            futures.append(
                                executor.submit(copy_file, filepath, data, tmp)
                            )
                            tmp_space_left -= size
                        else:
                            final_filepaths.append(filepath)
                    for future in concurrent.futures.as_completed(futures):
                        progress_bar.update(1)
                        final_filepaths.append(future.result())
                import multiprocessing

                # pool = multiprocessing.Pool()
                # results = []
                # for filepath, size in files_and_sizes.items():
                #     if size < tmp_space_left:
                #         args = (filepath, data, tmp)
                #         results.append(pool.apply_async(copy_file_parallel, args=(args,)))
                #         tmp_space_left -= size
                #     elif size < shm_space_left:
                #         args = (filepath, data, shm)
                #         results.append(pool.apply_async(copy_file_parallel, args=(args,)))
                #         shm_space_left -= size
                #     else:
                #         final_filepaths.append(filepath)
            # step 5: create a symlink to the data in /tmp, preserving the original folder structure in train_data
            for filepath in final_filepaths:
                if not filepath.startswith(tmp):
                    # get everything past os.path.basename(data) and prepend it to the filepath in /tmp
                    # For example, if we have filepath = /dev/shm/SST_256x256_6hrs_2012.zarr/Theta/0.0.595,
                    # we want to make a symlink at /tmp/SST_256x256_6hrs_2012.zarr/Theta/0.0.595
                    target_path = os.path.join(
                        tmp,
                        os.path.basename(data),
                        os.path.basename(filepath),
                    )
                    try:
                        os.symlink(filepath, target_path)
                        print(f"Creating symlink from {filepath} to {target_path}")
                    except Exception as e:
                        print(f"Encountered {e} when creating symlink from {data}.")
                        pass
            if is_train:
                self.args.data_path = os.path.join(tmp, os.path.basename(data))
            else:
                self.args.val_data_path = os.path.join(tmp, os.path.basename(data))

        else:
            print(
                "Not enough space on /tmp or /dev/shm to copy training data\n"
                f"(Total space on /tmp and /dev/shm: {(tmp_space + shm_space)/1e9:.2f} GB)"
                f"(Size of training data: {data_size/1e9:.2f} GB)"
            )


def get_size(path):
    if os.path.isfile(path):
        total_size = os.path.getsize(path)
    elif os.path.isdir(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
    else:
        total_size = -1
    return total_size


def copy_file(filepath, data_dir, root):
    target_path = os.path.join(
        root, os.path.basename(data_dir), os.path.relpath(filepath, data_dir)
    )
    # print(f"Copying {filepath} to {target_path}")
    if not os.path.exists(os.path.dirname(target_path)):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy(filepath, target_path)
    return target_path


def copy_file_parallel(filepath, data_dir, root):
    target_path = os.path.join(
        root, os.path.basename(data_dir), os.path.relpath(filepath, data_dir)
    )
    print(f"Copying {filepath} to {target_path}")
    if not os.path.exists(os.path.dirname(target_path)):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy(filepath, target_path)
    return target_path

def main():
    args = parse_args()
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)
    # On lonstar, specifying gpus_per_node gives:
    # sbatch: error: Batch job submission failed: Invalid generic resource (gres) specification
    executor.update_parameters(
        name=args.slurm_job_name,
        slurm_partition=args.slurm_partition,
        timeout_min=args.walltime,
        nodes=args.nodes,
        # gpus_per_node=args.ngpus_per_node,  # comment this out on Lonestar
        tasks_per_node=args.ngpus_per_node,  # one task per GPU
        slurm_comment=args.slurm_comment,
    )

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
