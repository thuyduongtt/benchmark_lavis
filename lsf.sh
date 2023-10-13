#!/bin/bash
# This script describes a typical LSF job script. Comments will explain what the
# specific option does.  All lines starting with # are ignored. LSF commands are
# prefixed by #BSUB and they must start a line. All #BSUB directives after the
# first non-comment line are ignored. To comment out a directive, simply add
# another hash (e.g. ##BSUB)
#
# Give the job a meaningful name.
#BSUB -J KAT
#
# Determine where the output will be written. WARNING: if you forget to specify
# this or if the directory does not exist, LSF will not create any output file.
#BSUB -o %x.%j.out
#
# The same goes for the error file. Make sure that "/home/${USER}/outdir"
# exists.
#BSUB -e %x.%j.err
#
# Specify the number of GPUs to be used per host.
#BSUB -gpu "num=2"
#
# Specify the type of GPU to use. Those are GeForce for rb_regular and Voltas
# for rb_highend.
#BSUB -q short
# #BSUB -q rb_highend       # use VOLTAS if uncommented
#
# Specify the number of job slots to be used. By default (but not for this
# example), this is also the number of CPUs.
#BSUB -n 2
#
# Specify the number of CPU cores per job slot. A job slot is always guaranteed
# to be on one host. Here, specify 3. The default is 1 CPU per job slot.
#BSUB -R "affinity[core(3)]"
#
# Receive email notifications.  Specify when to receive emails.
#BSUB -B			    # receive email when job starts
#BSUB -N			    # receive email when job ends
#
# Specify RAM PER JOB SLOT in MB that your job will use.
#BSUB -M 5000
#
# Specify the maximum runtime of your job. The format is "hours:minutes". The
# maximum runtime is 7 days at the moment. Here, we use 15min.
#BSUB -W 00:30
#
# Make sure that all job slots run on the same host
#BSUB -R "span[hosts=1]"    # run on a single host
# 
# Instead of span[hosts=1], you can also specify the number of job slots per
# host the following way (please remember that you specify the number of GPUs
# per host and not per job slot!). Note that only explicitly MPI-enabled jobs
# may run on multiple hosts at the same time. Without MPI, resources not on the
# primary host will simply sit idle and thus be wasted.
# #BSUB -R "span[ptile=2]"    # 2 job slots per host

# For members of BCAI-R:
# Please work in /fs/scratch/rng_cr_bcai_dl/$USER or your home directory. Also
# access your data directly from /fs/scratch/rng_cr_bcai_dl .

. /fs/applications/lsf/latest/conf/profile.lsf  # this line is needed
. /fs/applications/modules/current/init/bash    # this line is neeed

# Exit on any error. Do not put this line before the two needed ones.
set -e
# Unload all modules.
module purge
# Load a specific version of CUDA and CUDNN (or any other available version,
# check "module avail" for which ones there are).
module load cudnn
# Make conda available. We hard-code the version in this example.
module load conda

# Do stuff. Describe how to perform your experiment or simulation here.
conda activate kosmos2
python start.py --path_to_ds unbalanced --output_dir_name output_unbalanced

# No longer exit on any error.
set +e
