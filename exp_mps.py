import pdb
import time
import os
user = os.environ.get('USER')
import random
import json
import numpy as np
import glob
import argparse
import math
from pathlib import Path
import sys
import subprocess

# Import MISO config to get repository root
from miso_config import REPO_ROOT, get_path
sys.path.append(get_path('mps', 'scheduler', 'simulator'))
from utils import *
import copy
from controller_helper import *
import threading
import _thread
from exp_full import Experiment
sys.path.append(get_path('workloads'))
from send_signal import send_signal
import socket
from threading import Event

class MPS(Experiment):

    def __init__(self, args, physical_nodes, max_tenants=3):
        super().__init__(args, physical_nodes)
        self.tc = 'mps'
        self.max_tenants = max_tenants
        self.gpu_states = []
        for i in range(args.num_gpu):
            self.gpu_states.append(MPS_GPU_Status(i, max_tenants=max_tenants))        

    def _transfer_telemetry_files(self, run_log):
        """Transfer telemetry files from server to client logs/mps/ directory."""
        import os
        user = os.environ.get('USER')
        server_telemetry_dir = f'/scratch/{user}/telemetry'
        client_telemetry_dir = 'logs/mps'
        
        # Ensure client directory exists
        Path(client_telemetry_dir).mkdir(parents=True, exist_ok=True)
        
        for real_node in self.node_list:
            try:
                # Determine if we need SSH or can use direct path
                # If node is localhost, files might already be accessible
                if real_node == 'localhost' or real_node == '127.0.0.1':
                    # For localhost, try direct copy first
                    if os.path.exists(server_telemetry_dir):
                        import shutil
                        for file in Path(server_telemetry_dir).glob('*'):
                            if file.is_file():
                                dest = Path(client_telemetry_dir) / file.name
                                shutil.copy2(file, dest)
                                print(f'Copied telemetry file: {file.name}', file=run_log, flush=True)
                    else:
                        print(f'Warning: Telemetry directory {server_telemetry_dir} not found locally', file=run_log, flush=True)
                else:
                    # For remote nodes, use scp
                    # Try to determine username from node or use current user
                    # Format: user@host or just host
                    if '@' in real_node:
                        ssh_target = real_node
                    else:
                        ssh_target = f'{user}@{real_node}'
                    
                    # Transfer all telemetry files
                    scp_cmd = [
                        'scp',
                        '-q',  # Quiet mode
                        '-o', 'StrictHostKeyChecking=no',
                        '-o', 'UserKnownHostsFile=/dev/null',
                        f'{ssh_target}:{server_telemetry_dir}/*',
                        client_telemetry_dir + '/'
                    ]
                    
                    try:
                        result = subprocess.run(
                            scp_cmd,
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if result.returncode == 0:
                            print(f'Transferred telemetry files from {real_node} to {client_telemetry_dir}/', file=run_log, flush=True)
                        else:
                            print(f'Warning: scp failed for {real_node}: {result.stderr}', file=run_log, flush=True)
                            # Try alternative: use rsync if available
                            rsync_cmd = [
                                'rsync',
                                '-avz',
                                '-e', 'ssh -o StrictHostKeyChecking=no',
                                f'{ssh_target}:{server_telemetry_dir}/',
                                client_telemetry_dir + '/'
                            ]
                            result2 = subprocess.run(
                                rsync_cmd,
                                capture_output=True,
                                text=True,
                                timeout=30
                            )
                            if result2.returncode == 0:
                                print(f'Transferred telemetry files via rsync from {real_node}', file=run_log, flush=True)
                            else:
                                print(f'Warning: Both scp and rsync failed for {real_node}. Files may be on server at {server_telemetry_dir}', file=run_log, flush=True)
                    except subprocess.TimeoutExpired:
                        print(f'Warning: Timeout transferring telemetry files from {real_node}', file=run_log, flush=True)
                    except Exception as e:
                        print(f'Warning: Error transferring telemetry files from {real_node}: {e}', file=run_log, flush=True)
                        print(f'  Telemetry files may be available on server at {server_telemetry_dir}', file=run_log, flush=True)
            except Exception as e:
                print(f'Warning: Error in telemetry file transfer for {real_node}: {e}', file=run_log, flush=True)

    # try to schedule job on a list of GPUs
    def try_schedule(self, job, gpu_list, migration, run_log, mps_lvl):
        sched_done = False
        
        avail_gpu = [g for g in gpu_list if not g.full]
        if len(avail_gpu) > 0:
            gpu = avail_gpu[0]
            gpu.jobs.append(job)
            sched_done = True
            gpuid = self.gpu_states.index(gpu)
            real_node, real_gpu = self.GPU_LUT(gpuid)
            mps_start(real_node, job, real_gpu, level=mps_lvl, port=self.gpu_server_port)
            self.job_exe[job] = (gpuid, 0)
            print(f'Schedule time: {int(time.time()-self.start_time)}', file=run_log, flush=True)
            print(f'job {job} scheduled on GPU {gpu.index}, {real_node} device {real_gpu}', file=run_log, flush=True)
        return sched_done            

    def run(self, args, mps_lvl=33): 
        run_log = open('logs/experiment_mps.log','w')

        ####### start GPU telemetry collection on server (DCGM + NVML) ##########
        self.telemetry_enabled = False
        if hasattr(args, 'collect_telemetry') and args.collect_telemetry:
            try:
                # Get list of GPU IDs to monitor (0 to num_gpu-1)
                gpu_ids = list(range(args.num_gpu))
                sample_interval = getattr(args, 'telemetry_interval', 1.0)
                # Telemetry runs on server, output to server's scratch directory
                import os
                user = os.environ.get('USER')
                output_dir = f'/scratch/{user}/telemetry'
                
                # Start telemetry on GPU server
                for real_node in self.node_list:
                    start_telemetry(real_node, gpu_ids, sample_interval, output_dir, port=self.gpu_server_port)
                
                self.telemetry_enabled = True
                print(f'GPU telemetry collection started on server (DCGM+NVML, interval={sample_interval}s, output={output_dir})', file=run_log, flush=True)
            except Exception as e:
                print(f'Warning: Failed to start GPU telemetry on server: {e}', file=run_log, flush=True)
                print(f'  Ensure DCGM CLI (dcgmi) is installed on server and pynvml is available', file=run_log, flush=True)
                self.telemetry_enabled = False

        ####### start job listener ##########
        stop_event = Event()
        x = threading.Thread(target=thread_func, daemon=True, args=(stop_event, self, run_log, 'mps'))
        x.start()
        # Give thread a moment to start and bind to port
        time.sleep(1)
        # Check if thread is still alive (if it crashed, it won't be)
        if not x.is_alive():
            print('ERROR: Job listener thread failed to start!', file=run_log, flush=True)
            print('  Check logs above for port binding errors. Experiment cannot continue without listener.', file=run_log, flush=True)
            raise RuntimeError('Job listener thread failed to start - cannot receive job status updates')

        ####### initialize all GPUs #########
        for real_node in self.node_list:
            kill_all(real_node, port=self.gpu_server_port)
            broadcast_host(real_node, self, port=self.gpu_server_port)
        time.sleep(10)
        for gpu in self.gpu_states:
            real_node, real_gpu = self.GPU_LUT(gpu.index)
            start_mps(real_node, real_gpu, port=self.gpu_server_port)
            # config_gpu(real_node, real_gpu, 0)

        ###### initialize some variables ########
        queue = list(self.queue_dict)
        queue_ind = 0
    
        active_jobs_per_gpu = [] # time series of total number of jobs running
        arrived_jobs = []
        progress = {}
        migration  = {}
        for j in self.job_runtime:
            migration[j] = 0
        time.sleep(10)
        
        ##### start running ###########
        self.start_time = int(time.time())
        progress_time = int(time.time())

        while True:
            passed_time = int(time.time() - self.start_time)
#            if passed_time >= 900:
#                pdb.set_trace()
            while queue_ind < len(queue) and self.queue_dict[queue[queue_ind]] <= passed_time:
                arrived_jobs.append(queue[queue_ind])
                self.arrive_time[queue[queue_ind]] = int(time.time())
                queue_ind += 1

            if len(arrived_jobs) >= 1:
                '''
                priority
                1. there is idle GPU, so no migration at all
                2. least number of current jobs currently running, randomly pick one, order does not matter
                '''
                for job in arrived_jobs[:]:
                    sched_done = self.try_schedule(job, self.gpu_states, migration, run_log, mps_lvl)
                    if sched_done:
                        arrived_jobs.pop(0)
                        self.sched_time[job] = int(time.time())
                    else: # stop scheduling jobs, follow a strict FIFO pattern
                        break

            ############### wait for next iteration, job is running ##########
        
            self.emptied_gpu = {}
            time.sleep(args.step)            

            if int(time.time() - progress_time) >= 60:
                progress[int(time.time() - self.start_time)] = sum(list(self.completion.values()))
                progress_time = int(time.time())
             
            curr_time = int(time.time())
            emptied_list = []
            for gpu, emp_time in self.emptied_gpu.items():
                if curr_time - emp_time > 3: # give it 3 seconds to breath
                    emptied_list.append(gpu)
       
#            # first see if jobs in arrived_jobs can be scheduled on emptied gpus
            for job in arrived_jobs[:]:
                sched_done = self.try_schedule(job, emptied_list, migration, run_log, mps_lvl)
                if sched_done:
                    arrived_jobs.pop(0)
                    self.sched_time[job] = int(time.time())
                else: # stop scheduling jobs, follow a strict FIFO pattern
                    break

#            if no more arrived jobs can schedule, repartition emptied gpus:
            cnt_active = 0
            for gpu in self.gpu_states:
                cnt_active += len(gpu.active_jobs)
            active_jobs_per_gpu.append(cnt_active / args.num_gpu)
       
            # MPS jobs cannot calculate overall rate like this
            # self.overall_rate.append(sum([self.get_rate(gpu) for gpu in self.gpu_states]))
        
#            # sanity check
            for gpu in self.gpu_states:
                if len(gpu.jobs) > self.max_tenants:
                    raise RuntimeError(f'Check failed: GPU should not have >{self.max_tenants} jobs')
            
            ################ check if termination condition is met ################
        
            if sum(self.finish.values()) == len(self.finish) and queue_ind == args.num_job and len(arrived_jobs) == 0:
                print(f'Time: {int(time.time()-self.start_time)}, all jobs are finished!', file=run_log, flush=True)
                self.span_time = int(time.time()-self.start_time)
                self.overall_rate.append(self.span_time)
                break
            elif int(time.time()-self.start_time) >= 36000:
                print(f'WARNING: Experiment timeout after 36000 seconds. Jobs may not have completed.', file=run_log, flush=True)
                print(f'  Finished jobs: {sum(self.finish.values())}/{len(self.finish)}', file=run_log, flush=True)
                print(f'  Queue index: {queue_ind}/{args.num_job}', file=run_log, flush=True)
                print(f'  Arrived jobs waiting: {len(arrived_jobs)}', file=run_log, flush=True)
                # Break instead of pdb to allow cleanup
                break
       
        ########################
        Path('logs/mps').mkdir(parents=True, exist_ok=True)
        JCT, JRT, QT = {}, {}, {}
        
        for job in self.job_runtime:
            JCT[job] = self.comp_time[job] - self.arrive_time[job]
            QT[job] = self.sched_time[job] - self.arrive_time[job]
            JRT[job] = self.comp_time[job] - self.sched_time[job]
        
        for metric, name in zip([JCT, JRT, QT], ['JCT', 'JRT', 'QT']):
            metric['average'] = np.mean(list(metric.values()))
            with open(f'logs/mps/{name}.json', 'w') as f:
                json.dump(metric, f, indent=4)
        migration['average'] = np.mean(list(migration.values()))
        
        with open('logs/mps/active_jobs_per_gpu.json', 'w') as f:
            json.dump(active_jobs_per_gpu, f, indent=4)
        with open('logs/mps/completion.json', 'w') as f:
            json.dump(self.completion, f, indent=4)
        with open('logs/mps/progress.json', 'w') as f:
            json.dump(progress, f, indent=4)
        with open('logs/mps/migration.json', 'w') as f:
            json.dump(migration, f, indent=4)
        with open('logs/mps/ckpt_dict.json', 'w') as f:
            json.dump(self.ckpt_dict, f, indent=4)
        with open('logs/mps/ckpt_ovhd.json', 'w') as f:
            json.dump(self.ckpt_ovhd, f, indent=4)
        with open('logs/mps/overall_rate.json', 'w') as f:
            json.dump(self.overall_rate, f, indent=4)

        self.term_thread()
        stop_event.set()
        
        ####### stop GPU telemetry collection on server and transfer files ##########
        if hasattr(self, 'telemetry_enabled') and self.telemetry_enabled:
            try:
                for real_node in self.node_list:
                    stop_telemetry(real_node, port=self.gpu_server_port)
                print('GPU telemetry collection stopped on server', file=run_log, flush=True)
                
                # Transfer telemetry files from server to client
                self._transfer_telemetry_files(run_log)
            except Exception as e:
                print(f'Warning: Error stopping GPU telemetry on server: {e}', file=run_log, flush=True)
        
#        print('trying to join threads')    
#        x.join()
        print('done')
