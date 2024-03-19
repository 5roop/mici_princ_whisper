# mici_princ_whisper
Finetuning whisper on Mići Princ

# 2024-03-12T13:55:22

Figured out how to run stuff on SLURM, but ATM it's broken. In the past, the following worked:
```bash
srun --gpus=1 --time=01:01:00 --mem=10000 python 000_testing.py
```
but now it returns the following error:
```
srun: error: Unable to confirm allocation for job 911: Invalid job id specified
srun: Check SLURM_JOB_ID environment variable. Expired or invalid job 911
```

~Maybe it's due to the network problems JSI has been experiencing today.~

Barbara fixed the problem after an email report.


# 2024-03-19T08:52:39

The second epoch was chosen as the optimal. In this case