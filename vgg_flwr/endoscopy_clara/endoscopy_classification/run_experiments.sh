#!/usr/bin/env bash

# download dataset
python3 ./pt/utils/endoscopy_download_data.py

# central
./submit_job.sh endoscopy_central 0.0

# FedAvg
./submit_job.sh endoscopy_fedavg 1.0
./submit_job.sh endoscopy_fedavg 0.5
./submit_job.sh endoscopy_fedavg 0.3
./submit_job.sh endoscopy_fedavg 0.1

# FedProx
./submit_job.sh endoscopy_fedprox 0.1

# FedOpt
./submit_job.sh endoscopy_fedopt 0.1

# SCAFFOLD
./submit_job.sh endoscopy_scaffold 0.1

# FedAvg + HE
./submit_job.sh endoscopy_fedavg_he 1.0

# FedAvg with TensorBoard streaming
./submit_job.sh endoscopy_fedavg_stream_tb 1.0
