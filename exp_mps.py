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
            schedule_msg = f'Schedule time: {int(time.time()-self.start_time)}, job {job} scheduled on GPU {gpu.index}, {real_node} device {real_gpu}'
            print(schedule_msg, file=run_log, flush=True)
            print(schedule_msg, flush=True)  # Also print to stdout for immediate visibility
        return sched_done            

    def run(self, args, mps_lvl=33): 
        run_log = open('logs/experiment_mps.log','w')
        
        # Initialize wandb if enabled
        wandb_run = None
        if hasattr(args, 'use_wandb') and args.use_wandb:
            try:
                import wandb
                # Generate run name if not provided
                run_name = args.wandb_run_name
                if not run_name:
                    run_name = f"mps_lvl{mps_lvl}_jobs{args.num_job}_gpus{args.num_gpu}_seed{args.seed}"
                
                # Initialize wandb
                wandb.init(
                    project=getattr(args, 'wandb_project', 'mps-experiments'),
                    entity=getattr(args, 'wandb_entity', None),
                    name=run_name,
                    tags=getattr(args, 'wandb_tags', []),
                    config={
                        'mps_level': mps_lvl,
                        'num_job': args.num_job,
                        'num_gpu': args.num_gpu,
                        'arrival': args.arrival,
                        'seed': args.seed,
                        'step': args.step,
                        'max_tenants': self.max_tenants,
                        'filler': getattr(args, 'filler', False),
                        'flat_arrival': getattr(args, 'flat_arrival', False),
                        'random_trace': getattr(args, 'random_trace', False),
                        'collect_telemetry': getattr(args, 'collect_telemetry', False),
                        'telemetry_interval': getattr(args, 'telemetry_interval', 1.0),
                    }
                )
                wandb_run = wandb
                print(f'W&B logging enabled: project={wandb_run.config.get("_wandb", {}).get("project")}, run={run_name}', file=run_log, flush=True)
            except ImportError:
                print('Warning: wandb not installed. Install with: pip install wandb', file=run_log, flush=True)
            except Exception as e:
                print(f'Warning: Failed to initialize wandb: {e}', file=run_log, flush=True)

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
            
            # Diagnostic: Check if jobs are stuck (no PID received after reasonable time)
            stuck_jobs = []
            for job in list(self.job_exe.keys()):
                if self.job_exe[job][0] is not None:  # Job is scheduled
                    if self.pid_dict.get(job, 0) == 0:  # No PID received yet
                        time_since_scheduled = int(time.time()) - self.sched_time.get(job, self.start_time)
                        if time_since_scheduled > 30:  # 30 seconds without PID
                            stuck_jobs.append((job, time_since_scheduled))
                            # Print to BOTH log file AND stdout so user sees it immediately
                            warning_msg = (
                                f'\n⚠️  WARNING: Job {job} scheduled {time_since_scheduled}s ago but no PID received!\n'
                                f'   This usually means the workload cannot connect back to the scheduler.\n'
                                f'   Possible causes:\n'
                                f'     1. Job listener thread is not running (check port {self.gpu_server_port})\n'
                                f'     2. Reverse SSH tunnel not set up (workloads need: ssh -R {self.gpu_server_port}:localhost:{self.gpu_server_port} <server>)\n'
                                f'     3. Workload failed to start on server\n'
                                f'   Check server logs: /scratch/{user}/miso_logs/mps/job{job}_start.err\n'
                            )
                            print(warning_msg, file=run_log, flush=True)
                            print(warning_msg, flush=True)  # Also print to stdout
            
            # If jobs are stuck, provide immediate actionable feedback
            if stuck_jobs:
                print(f'\n🔴 STUCK JOBS DETECTED: {len(stuck_jobs)} job(s) waiting for PID:', flush=True)
                for job, wait_time in stuck_jobs:
                    print(f'   - Job {job}: waiting {wait_time}s for PID', flush=True)
                print(f'\n💡 QUICK FIXES:', flush=True)
                print(f'   1. Check if listener is running: lsof -i :{self.gpu_server_port}', flush=True)
                print(f'   2. Check server logs: ssh <server> "tail -f /scratch/{user}/miso_logs/mps/job{stuck_jobs[0][0]}_start.err"', flush=True)
                print(f'   3. Verify reverse tunnel: ssh -R {self.gpu_server_port}:localhost:{self.gpu_server_port} <server>', flush=True)
                print(f'   4. Check experiment log: tail -f logs/experiment_mps.log\n', flush=True)            

            if int(time.time() - progress_time) >= 60:
                progress[int(time.time() - self.start_time)] = sum(list(self.completion.values()))
                progress_time = int(time.time())
                # Log progress to wandb
                if wandb_run:
                    wandb_run.log({
                        'progress/completed_jobs': sum(list(self.completion.values())),
                        'progress/total_jobs': len(self.completion),
                        'progress/completion_rate': sum(list(self.completion.values())) / len(self.completion) if len(self.completion) > 0 else 0,
                        'time/elapsed_seconds': int(time.time() - self.start_time),
                    }, step=int(time.time() - self.start_time))
             
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
            
            # Log active jobs to wandb
            if wandb_run:
                wandb_run.log({
                    'gpu/active_jobs_per_gpu': cnt_active / args.num_gpu,
                    'gpu/total_active_jobs': cnt_active,
                    'gpu/total_gpus': args.num_gpu,
                    'time/elapsed_seconds': int(time.time() - self.start_time),
                }, step=int(time.time() - self.start_time))
       
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
                # Check for stuck jobs
                stuck_jobs = [j for j in self.job_exe.keys() if self.pid_dict.get(j, 0) == 0 and self.job_exe[j][0] is not None]
                if stuck_jobs:
                    print(f'  Stuck jobs (no PID received): {stuck_jobs}', file=run_log, flush=True)
                    print(f'    These jobs may not be able to connect back to scheduler.', file=run_log, flush=True)
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
        
        # Log final metrics to wandb
        if wandb_run:
            # Log summary statistics
            wandb_run.log({
                'metrics/jct_mean': JCT['average'],
                'metrics/jct_std': np.std([v for k, v in JCT.items() if k != 'average']),
                'metrics/jrt_mean': JRT['average'],
                'metrics/jrt_std': np.std([v for k, v in JRT.items() if k != 'average']),
                'metrics/qt_mean': QT['average'],
                'metrics/qt_std': np.std([v for k, v in QT.items() if k != 'average']),
                'metrics/migration_mean': migration['average'],
                'metrics/migration_total': sum([v for k, v in migration.items() if k != 'average']),
                'metrics/makespan_seconds': self.span_time,
                'metrics/total_jobs': len(self.job_runtime),
                'metrics/completed_jobs': sum(list(self.completion.values())),
            })
            
            # Log time-series data as wandb tables
            try:
                # Active jobs per GPU time series
                import pandas as pd
                active_jobs_df = pd.DataFrame({
                    'step': range(len(active_jobs_per_gpu)),
                    'active_jobs_per_gpu': active_jobs_per_gpu
                })
                wandb_run.log({'active_jobs_per_gpu_table': wandb.Table(dataframe=active_jobs_df)})
                
                # Progress time series
                if progress:
                    progress_df = pd.DataFrame({
                        'time_seconds': list(progress.keys()),
                        'completed_jobs': list(progress.values())
                    })
                    wandb_run.log({'progress_table': wandb.Table(dataframe=progress_df)})
                
                # Per-job metrics
                jobs_df = pd.DataFrame({
                    'job_id': [k for k in JCT.keys() if k != 'average'],
                    'jct': [JCT[k] for k in JCT.keys() if k != 'average'],
                    'jrt': [JRT[k] for k in JRT.keys() if k != 'average'],
                    'qt': [QT[k] for k in QT.keys() if k != 'average'],
                    'migration': [migration[k] for k in migration.keys() if k != 'average'],
                })
                wandb_run.log({'per_job_metrics': wandb.Table(dataframe=jobs_df)})
            except ImportError:
                print('Warning: pandas not available, skipping wandb table logging', file=run_log, flush=True)
            
            # Log histograms
            try:
                jct_values = [v for k, v in JCT.items() if k != 'average']
                jrt_values = [v for k, v in JRT.items() if k != 'average']
                qt_values = [v for k, v in QT.items() if k != 'average']
                
                wandb_run.log({
                    'histograms/jct': wandb.Histogram(jct_values),
                    'histograms/jrt': wandb.Histogram(jrt_values),
                    'histograms/qt': wandb.Histogram(qt_values),
                })
            except Exception as e:
                print(f'Warning: Failed to log histograms to wandb: {e}', file=run_log, flush=True)
            
            # Finish wandb run
            wandb_run.finish()
            print('W&B logging completed', file=run_log, flush=True)

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
