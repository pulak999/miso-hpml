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

# Import MISO config to get repository root
from miso_config import REPO_ROOT, get_path
sys.path.append(get_path('mps', 'scheduler', 'simulator'))
from utils import *
import copy
sys.path.append(get_path('workloads'))
from send_signal import send_signal
import socket
import threading
import _thread
import signal
from tcp_interpreter import *
from threading import Event

# start job
def start_job(node, job, gpu, sliceid):
    cmd = f'start {job} gpu {gpu} slice {sliceid}'
    send_signal(node, cmd=cmd)   

def mps_start(node, job, gpu, level=50, port=10002):
    cmd = f'mps_strt {job} gpu {gpu} lvl {level}'
    send_signal(node, port=port, cmd=cmd)   

def mps_resume(node, job, gpu, resume_batch, level=50):
    cmd = f'mps_rsm {job} gpu {gpu} batch {resume_batch} lvl {level}'
    send_signal(node, cmd=cmd)   

def resume_job(node, job, gpu, sliceid, resume_batch):
    cmd = f'resume {job} gpu {gpu} slice {sliceid} batch {resume_batch}'
    send_signal(node, cmd=cmd)   

def config_gpu(node, gpu, partition):
    cmd = f'config gpu {gpu} partition {partition}'
    send_signal(node, cmd=cmd)

def start_mps(node, gpu, port=10002): # this is essentially reset_mig and create_ins 7g.40gb, then run ./enable_mps_on_mig
    cmd = f'mps_enable {gpu}'
    send_signal(node, port=port, cmd=cmd)

def end_mps(node, gpu): 
    cmd = f'mps_disable {gpu}'
    send_signal(node, cmd=cmd)

def fkill_job(node, job, pid):
    cmd = f'fkill {job} pid {pid}'
    send_signal(node, cmd=cmd)

def kill_all(node, port=10002):
    cmd = 'kill all'
    send_signal(node, port=port, cmd=cmd)

def broadcast_host(node, runtime, port=10002):
    # When using SSH tunnel (node == 'localhost'), workloads on remote server
    # need to connect back. Use 'localhost' so they connect to remote localhost,
    # which requires a reverse SSH tunnel to be set up.
    # For direct connections, use the actual hostname.
    if node == 'localhost' or node == '127.0.0.1':
        # Using SSH tunnel - workloads should connect to localhost on remote side
        # This requires reverse SSH tunnel: ssh -R 10002:localhost:10002 l4vm
        hostname_to_send = 'localhost'
    else:
        hostname_to_send = socket.gethostname()
    
    cmd = f'hostname {hostname_to_send}'
    send_signal(node, port=port, cmd=cmd)
    cmd = f'log_dir {runtime.tc}'
    send_signal(node, port=port, cmd=cmd)

def start_telemetry(node, gpu_ids, interval=1.0, output_dir=None, port=10002):
    """Start GPU telemetry collection on the server."""
    if output_dir is None:
        import os
        user = os.environ.get('USER')
        output_dir = f'/scratch/{user}/telemetry'
    gpu_ids_str = ','.join(str(gid) for gid in gpu_ids)
    cmd = f'telemetry_start {gpu_ids_str} {interval} {output_dir}'
    send_signal(node, port=port, cmd=cmd)

def stop_telemetry(node, port=10002):
    """Stop GPU telemetry collection on the server."""
    cmd = 'telemetry_stop'
    send_signal(node, port=port, cmd=cmd)

def save_jobs(node, job_list, runtime, run_log):
    finish_status = [runtime.finish[job] for job in job_list]
    if 1 in finish_status:
        print(f'checkpoing {job_list} is invalid due to finish status {finish_status}', file=run_log, flush=True) #TODO: see if something can be done for jobs that are near finishing
        return False # indicate this save is invalid
    for job in job_list:
        if runtime.pid_dict[job] == 0:
            raise RuntimeError(f'job {job} PID is not updated')
        cmd = f'save {job} pid {runtime.pid_dict[job]}'
        send_signal(node, cmd=cmd)
    # wait till all ckpt signals are received
    print(f'waiting for checkpoint to finish, jobs {str(job_list)}', file=run_log, flush=True)
    while True:        
        # make sure all ckpt_dict are 1
        time.sleep(0.5) #TODO: 1 pdb.set_trace()
        ckpt_sum = 0
        for job in job_list:
            ckpt_sum += runtime.ckpt_dict[job]      
        if ckpt_sum == len(job_list):
            print('checkpoint finished', file=run_log, flush=True) 
            for job in job_list:
                fkill_job(node, job, runtime.pid_dict[job])
            return True

def thread_func(event, runtime, run_log, mode='full'): # this is an instance of the Experiment class 
    # here listen on the socket 
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Allow reuse of address to avoid "Address already in use" errors
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Use 'localhost' instead of gethostname() to avoid resolution issues, especially with SSH tunnels
    # Workloads connect to 'localhost' on remote side, which tunnels back via reverse SSH tunnel
    server_address = ('localhost', 10002)
    print('starting up on {} port {}'.format(*server_address), file=run_log, flush=True)
    
    # Try to bind, with retry logic
    max_retries = 5
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            sock.bind(server_address)
            break  # Success, exit retry loop
        except OSError as e:
            if e.errno == 48:  # Address already in use
                if attempt < max_retries - 1:
                    print(f'Port 10002 in use, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})...', file=run_log, flush=True)
                    time.sleep(retry_delay)
                    # Close and recreate socket for retry
                    try:
                        sock.close()
                    except:
                        pass
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                else:
                    print(f'ERROR: Port 10002 is already in use after {max_retries} attempts.', file=run_log, flush=True)
                    print(f'  This usually means:', file=run_log, flush=True)
                    print(f'  1. A previous experiment is still running', file=run_log, flush=True)
                    print(f'  2. A previous experiment did not clean up properly', file=run_log, flush=True)
                    print(f'  Solution: Kill any existing processes using port 10002:', file=run_log, flush=True)
                    print(f'    lsof -ti:10002 | xargs kill -9', file=run_log, flush=True)
                    print(f'  Or wait a few seconds for the port to be released', file=run_log, flush=True)
                    raise
            else:
                raise
    
    sock.listen(5) 

    while not event.is_set():
        # Wait for a connection
        connection, client_address = sock.accept()      
        try:
            while not event.is_set():
                data = connection.recv(32)
                if data: 
                    data_str = data.decode('utf-8')
                    if 'term_thread' in data_str:                        
                        connection.sendall(b'success')
                        print('breaking connection loop')
                        sys.exit()
                        # connection.close()
                    elif mode == 'full':
                        interpret_full(data_str, runtime, run_log)
                    elif mode == 'mps':
                        interpret_mps(data_str, runtime, run_log)                        
                    elif mode == 'static':
                        interpret_static(data_str, runtime, run_log)
                    elif mode == 'oracle' or mode == 'miso':
                        interpret_miso(data_str, runtime, run_log)

#                    elif 'waste' in data_str:
#                        global epoch_waste_dict
#                        job_name = data_str.split(' ')[0]
#                        epoch_waste_time = data_str.split(' ')[2]
#                        epoch_waste_dict[job_name] += int(epoch_waste_time)
#                    elif 'b_end' in data_str:
#                        job_name = data_str.split(' ')[0]
#                        job = job_name.replace('job','')
#                        ovhd_b[job].append(int(time.time() - b_start[job]))
#                        c_start[job] = time.time()
#                    elif 'c_end' in data_str:
#                        job_name = data_str.split(' ')[0]
#                        job = job_name.replace('job','')
#                        ovhd_c[job].append(int(time.time() - c_start[job]))
#                        d_start[job] = time.time()
#                    elif 'd_end' in data_str:
#                        job_name = data_str.split(' ')[0]
#                        job = job_name.replace('job','')
#                        ovhd_d[job].append(int(time.time() - d_start[job]))
#                        ovhd_total[job].append(int(time.time() - ovhd_start[job]))
#                        if ovhd_start[job] != 0:
#                            overhead[job] += int(time.time() - ovhd_start[job])
#                            ovhd_start[job] = 0 
#                            if job in list(K80_job.values()):
#                                K80_start_time[job] = time.time()
#                            elif job in list(V100_job.values()):
#                                V100_start_time[job] = time.time()
#                                promote_start_time[job] = time.time()
#                    elif '1st_epoch' in data_str: # 'job50 1st_epoch 35'
#                        job_name = data_str.split(' ')[0]
#                        job = job_name.replace('job','')
#                        epoch_time = int(data_str.split(' ')[2])
#                        if job in list(K80_job.values()):
#                            k80_1st[job].append(epoch_time)
#                        elif job in list(V100_job.values()):
#                            v100_1st[job].append(epoch_time)
                    #if 'ckpt_qual' in data_str or 'finish' in data_str or 'checkpoint' in data_str:
                    #    print('received ' + data_str)
                    connection.sendall(b'success')
                    #time.sleep(5)
                else:
                    break
        finally:
            connection.close()
            # print('Terminated connection')
    # Clean up socket when thread exits
    try:
        sock.close()
    except:
        pass
    print('Terminated thread')
