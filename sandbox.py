import pyopencl as cl 
import numpy as np 
from time import time
import os
from collections import Counter

def main():
	#fname = "~/Downloads/'Bioinformatics Python'/data/hg38.chroms/chr1.fa"
	fname = "E.ColiGenome.txt" #Toy example
	K = 1
	f = open(fname, 'r')
	#f.readline()
	data = f.read().upper().splitlines()
	f.close()
	data = ''.join(data).encode('utf-8')
	print(Counter(data))
	print(len(data))
	h_seq = np.frombuffer(data, dtype=np.uint8)
	h_seq = h_seq.astype(np.int32)
	h_seq = np.concatenate((np.zeros(4**K+1).astype(np.int32), h_seq))

	kernelsource = '''
	__kernel void mapToNumb(
		const int N,
		const int M,
		const int numbKmer,
		__global int* seq,
		__global int* numb_seq
	)
	{
		int gid = get_global_id(0);
		int idx = gid * M + numbKmer;
		int i, letter;

		if(gid < N) {
			for(i=0; i < M; i++) {
				letter = seq[idx+i];
				if(letter == 65) {
					numb_seq[idx+i] = 0;
				} else {
				if(letter == 67) {
					numb_seq[idx+i] = 1;
				} else {
				if(letter == 71) {
					numb_seq[idx+i] = 2;
				} else {
				if(letter == 84) {
					numb_seq[idx+i] = 3;
				} else {
					numb_seq[idx+i] = (-1) * numbKmer;
				}
				}
				}
				}
			}
		}
	}
	__kernel void freqTab(
		const int N,
		const int M,
		const int nK,
		const int numbKmer,
		__global int* numb_seq,
		__global int* freq_seq
	) {
		int gid = get_global_id(0);
		int idx = gid * M + numbKmer;
		int i, numb;
		int k, p, loc_idx, ptn_idx;
		for(i=0; i < M; i++) {
			ptn_idx = 1;
			loc_idx = idx + i;
			if(loc_idx <= (N*M + numbKmer - nK)) {
				for(k=0; k < nK; k++) {
					numb = numb_seq[loc_idx + k];
					for(p=nK-1-k; p > 0 ; p--) {
						ptn_idx *= 4;
					}
					ptn_idx = ptn_idx * numb + 1;
				}
				if(ptn_idx >= 0) {
					atomic_inc(&freq_seq[ptn_idx]);
				} else {
					atomic_inc(&freq_seq[0]);
				}
			}
		}
	}
	'''

	context = cl.create_some_context()
	device = context.devices[0]

	work_group_size = device.max_work_group_size
	work_item_size = device.max_work_item_sizes[0]
	print(work_group_size)
	print(work_item_size)

	numbGroups = 1024
	numbItems = 64

	seqLen = np.size(h_seq)
	q = int(seqLen/(numbGroups*numbItems))
	if q == 0:
		r = (numbGroups*numbItems) - seqLen	
	else:
		r = seqLen % q
	if r != 0:
		h_seq = np.concatenate((h_seq, np.zeros(r).astype(np.int32)))
		q = q+1
	h_numb_seq = np.zeros(np.size(h_seq)).astype(np.int32)
	print(q)
	print(r)

	queue = cl.CommandQueue(context)
	program = cl.Program(context, kernelsource).build()
	mapToNumb = program.mapToNumb
	mapToNumb.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None, None])
	freqTab = program.freqTab
	freqTab.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, np.int32, None, None])

	d_seq = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = h_seq)
	d_numb_seq = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_numb_seq.nbytes)
	cl.enqueue_fill_buffer(queue, d_numb_seq, np.zeros(1).astype(np.int32), 0, h_numb_seq.nbytes)

	N = numbGroups*numbItems
	M = q
	numbKmer = 4**K + 1
	globalsize = (N,)
	localsize = (numbItems,)

	mapToNumb(queue, globalsize, None, N, M, numbKmer, d_seq, d_numb_seq)
	
	queue.finish()

	#cl.enqueue_copy(queue, h_numb_seq, d_numb_seq)
	#print(h_numb_seq[:20])

	#d_numb_seq = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = h_numb_seq)
	h_freq_seq = np.zeros(np.size(h_seq)).astype(np.int32)
	d_freq_seq = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_freq_seq.nbytes)
	cl.enqueue_fill_buffer(queue, d_freq_seq, np.zeros(1).astype(np.int32), 0, h_freq_seq.nbytes)

	freqTab(queue, globalsize, None, N, M, K, numbKmer, d_numb_seq, d_freq_seq)

	print("check point")

	queue.finish()

	cl.enqueue_copy(queue, h_freq_seq, d_freq_seq)

	print("Mapping Done")
	print(h_freq_seq[:numbKmer])
	#print(Counter(data))

if __name__=="__main__":
	main()