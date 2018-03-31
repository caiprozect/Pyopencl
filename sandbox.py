import pyopencl as cl 
import numpy as np 
from time import time
import os
from collections import Counter

os.environ['CL_LOG_ERRORS']='stdout'

def main():
	#fname = "../data/chroms/chr1.fa"
	fname = "E.ColiGenome.txt" #Toy example
	K = 2
	f = open(fname, 'r')
	#f.readline()
	data = f.read().upper().splitlines()
	#print(Counter(data))
	f.close()
	data = ''.join(data).encode('utf-8')
	h_seq = np.frombuffer(data, dtype=np.uint8)
	h_seq = h_seq.astype(np.int)
	h_seq = np.concatenate((np.empty(4**K).astype(np.int), h_seq))

	kernelsource = '''
	__kernel void mapToNumb(
		const int N,
		const int W,
		const int M,
		const int K,
		__global int* seq,
		__global int* numb_seq
	)
	{
		int gid = get_global_id(0);
		int idx = gid * M;
		int i, letter;

		if(gid < N) {
			for(i=0; i < M; i++) {
				letter = seq[idx+i];
				if(letter == 65){
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
					numb_seq[idx+i] = -(int)(pow((float)4, (float)K));
				}
				}
				}
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

	#For GTX 970M
	numbGroups = 1024
	numbItems = 64

	seqLen = np.size(h_seq)
	q = int(seqLen/(numbGroups*numbItems))
	r = seqLen % q
	if r != 0:
		h_seq = np.concatenate((h_seq, np.zeros(r).astype(np.int)))
		print(h_seq)
		q = q+1
	h_numb_seq = np.repeat(0, np.size(h_seq)).astype(np.int)
	print(q)
	print(r)

	queue = cl.CommandQueue(context)
	program = cl.Program(context, kernelsource).build()
	mapToNumb = program.mapToNumb
	mapToNumb.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, np.int32, None, None])

	d_seq = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = h_seq)
	d_numb_seq = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_numb_seq.nbytes)

	N = numbGroups*numbItems
	W = numbItems
	M = q
	globalsize = (N,)
	localsize = (numbItems,)

	mapToNumb(queue, globalsize, None, N, W, M, K, d_seq, d_numb_seq)

	queue.finish()

	print("check point")

	cl.enqueue_copy(queue, h_numb_seq, d_numb_seq)
	print("Mapping Done")
	print(h_numb_seq[:20])
	#print(Counter(h_numb_seq))

if __name__=="__main__":
	main()