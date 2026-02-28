# Last updated on: 16 Sep 2023

import dpdata
import numpy as np

main_dir = "all-frames"
outdir = "dpdata/"
num_jobs=1

ms2 = dpdata.MultiSystems()

tot_frames = 0
## Jobs 1 to ... ##
for d in range(1,num_jobs+1):
	this_dir = main_dir + "/job{}/".format(d)
	fname = "{}/run-cp2k.log".format(this_dir)
	job_inds = np.unique(np.loadtxt(fname,dtype="int32"))
	nframes = job_inds.shape[0]
	print(nframes)
	print("JOB {} -- {} frames ".format(d,nframes))

	for j in job_inds:
		spelogname = "{}/{}/spe.log".format(this_dir,j)
		ls = dpdata.LabeledSystem(spelogname,fmt="cp2k/output")
		print(j)
		print(ls["energies"])
		ms2.append(ls)
		#print("loaded frame {}".format(j))

	tot_frames += nframes

print("Total number of frames from cp2k.log: {}".format(tot_frames))

for k in ms2.systems.keys():
	print("Systems: ",k)

## Shuffle the data
ms2.systems[k].shuffle()
#print(ms2.systems)
tot_dp_frames = ms2.systems[k].get_nframes()
print("Total number of frames from dpdata: {}".format(tot_dp_frames))

if tot_dp_frames < tot_frames:
	print("WARN: some spe.log files might be corrupted!")

## split into train/val -- 90:10
frac_train = 0.90
indx_train = np.arange(0,int(frac_train*tot_dp_frames),1,dtype=int)
num_train = indx_train.shape[0]
indx_val = np.arange(indx_train[-1]+1,tot_dp_frames,1,dtype=int)
num_val = indx_val.shape[0]

data_train = ms2.systems[k].sub_system(indx_train)
data_val = ms2.systems[k].sub_system(indx_val)

data_train.to_deepmd_raw(outdir+"train/raw/")
data_train.to_deepmd_npy(outdir+"train/npy/")

data_val.to_deepmd_raw(outdir+"val/raw/")
data_val.to_deepmd_npy(outdir+"val/npy/")
