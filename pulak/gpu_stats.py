### needs to run in the background
### needs to save data to a specifc folder, that we set inside the script

#!/usr/bin/env python3
import time
import csv
import dcgm_agent
import dcgm_fields
import dcgm_structs
import pydcgm


DEVICE_FIELDS = [
    dcgm_fields.DCGM_FI_DEV_GPU_UTIL,
    dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL,
    dcgm_fields.DCGM_FI_DEV_FB_USED,
    dcgm_fields.DCGM_FI_DEV_FB_TOTAL,
    dcgm_fields.DCGM_FI_DEV_POWER_USAGE,
    dcgm_fields.DCGM_FI_DEV_GPU_TEMP
]

PROCESS_FIELDS = [
    dcgm_fields.DCGM_FI_PROF_SM_ACTIVE,
    dcgm_fields.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE,
    dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE,
    dcgm_fields.DCGM_FI_DEV_FB_USED,
]


def write_csv_header(file, fields):
    writer = csv.writer(file)
    writer.writerow(["timestamp", "gpuId", "pid"] + fields)


def main():
    # Init DCGM
    dcgm_agent.dcgmInit()
    handle = dcgm_agent.dcgmStartEmbedded(dcgm_structs.DCGM_OPERATION_MODE_AUTO)

    # ----- Create GPU group -----
    gpu_group = pydcgm.DcgmGroup(handle, groupName="allGpus")
    gpu_group.AddAllGpus()

    # ----- Create FieldGroups -----
    device_fg = pydcgm.DcgmFieldGroup(handle, "deviceFields", DEVICE_FIELDS)
    process_fg = pydcgm.DcgmFieldGroup(handle, "processFields", PROCESS_FIELDS)

    # ----- Create Monitors -----
    device_mon = pydcgm.DcgmMonitor(handle, gpu_group, device_fg, 1000000)
    process_mon = pydcgm.DcgmProcess(handle, process_fg)

    # ----- Open CSV files -----
    dev_file = open("gpu_metrics.csv", "w", newline="")
    proc_file = open("process_metrics.csv", "w", newline="")

    write_csv_header(dev_file, [dcgm_fields.DCGM_FI_TO_NAME[f] for f in DEVICE_FIELDS])
    write_csv_header(proc_file, [dcgm_fields.DCGM_FI_TO_NAME[f] for f in PROCESS_FIELDS])

    dev_writer = csv.writer(dev_file)
    proc_writer = csv.writer(proc_file)

    print("Collecting GPU + process metrics... Press Ctrl+C to stop.")

    while True:
        timestamp = int(time.time())

        # ----- Device metrics -----
        device_vals = device_mon.GetLatestValues()
        for v in device_vals:
            row = [timestamp, v["gpuId"], ""]
            for f in DEVICE_FIELDS:
                row.append(v.get(f, 0))
            dev_writer.writerow(row)
            dev_file.flush()

        # ----- Per-process metrics -----
        process_vals = process_mon.GetAll()
        for pv in process_vals:
            row = [timestamp, pv.gpuId, pv.pid]
            for f in PROCESS_FIELDS:
                row.append(pv.fieldValues.get(f, 0))
            proc_writer.writerow(row)
            proc_file.flush()

        time.sleep(1)


if __name__ == "__main__":
    main()