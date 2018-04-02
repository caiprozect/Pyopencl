import pyopencl as cl 
import numpy as np 
from time import time
import os
from itertools import product
from functools import reduce
from operator import mul
from collections import Counter

os.environ["PYOPENCL_CTX"]=''
CPU_SIDE_INT = np.int32 #Change this according to architecture

def genSeq(file):
	print("Generate seq from {}".format(file))
	f = open(file, 'rb')
	f.readline()
	data = f.read().upper().splitlines()
	f.close()
	data = ("".encode('utf-8')).join(data)	
	seq = np.frombuffer(data, dtype=np.uint8).astype(CPU_SIDE_INT)
	print("Seq for {} has been generated".format(file))
	return seq

def kMerCount(file, nK):
	K = nK
	h_seq = genSeq(file)
	h_seq = np.concatenate((np.zeros(2+4+4**K).astype(CPU_SIDE_INT), h_seq))
	textLen = h_seq.size

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
		int idx = gid * M + numbKmer + 2 + 4;
		int i, letter;

		if(idx < N*M + numbKmer + 2 + 4) {
			for(i=0; i < M; i++) {
				letter = seq[idx+i];
				if(letter == 65) {
					numb_seq[idx+i] = 0;
					atomic_inc(&numb_seq[2]);
				} else {
				if(letter == 67) {
					numb_seq[idx+i] = 1;
					atomic_inc(&numb_seq[3]);
				} else {
				if(letter == 71) {
					numb_seq[idx+i] = 2;
					atomic_inc(&numb_seq[4]);
				} else {
				if(letter == 84) {
					numb_seq[idx+i] = 3;
					atomic_inc(&numb_seq[5]);
				} else {
				if(letter == 78) {
					numb_seq[idx+i] = (-1) * (numbKmer + 2 + 4);
				} else {
					numb_seq[idx+i] = (-1) * (numbKmer + 2 + 4) - 1000000;
				}
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
		__global int* numb_seq
	) {
		int gid = get_global_id(0);
		int idx = gid * M + numbKmer + 2 + 4;
		int i, numb;
		int k, p, loc_idx, ptn_idx;
		int dgt;
		for(i=0; i < M; i++) {
			ptn_idx = 0;
			loc_idx = idx + i;
			if(loc_idx <= (N*M + numbKmer + 2 + 4 - nK)) {
				for(k=0; k < nK; k++) {
					if((nK-1-k)==0) {
						dgt = 1;
					} else {
						dgt = (int)(pow((float)4, (float)(nK-1-k)));
					}
					numb = numb_seq[loc_idx + k];
					ptn_idx += dgt * numb;
				}
				ptn_idx += 2 + 4;
				if(ptn_idx >= 0) {
					atomic_inc(&numb_seq[ptn_idx]);
				} else {
				if(ptn_idx < -1000000) {
					atomic_inc(&numb_seq[0]);
				} else{
					atomic_inc(&numb_seq[1]);
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

	numbGroups = work_group_size
	numbItems = work_item_size

	seqLen = np.size(h_seq) - 4**K - 2 - 4
	q, r = divmod(seqLen, numbGroups*numbItems)
	q = q + 1
	h_seq = np.concatenate((h_seq, np.repeat(78, numbGroups*numbItems-r).astype(CPU_SIDE_INT)))
	h_numb_seq = np.zeros(h_seq.size).astype(CPU_SIDE_INT)
	print(q)
	print(numbGroups*numbItems-r)

	queue = cl.CommandQueue(context)
	program = cl.Program(context, kernelsource).build()
	mapToNumb = program.mapToNumb
	mapToNumb.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None, None])
	freqTab = program.freqTab
	freqTab.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, np.int32, None])

	d_seq = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = h_seq)
	d_numb_seq = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = h_numb_seq)
	#cl.enqueue_fill_buffer(queue, d_numb_seq, np.zeros(1).astype(np.int), 0, h_numb_seq.nbytes)

	N = numbGroups*numbItems
	M = q
	numbKmer = 4**K
	globalsize = (N,)
	localsize = (numbItems,)

	mapToNumb(queue, globalsize, None, N, M, numbKmer, d_seq, d_numb_seq)
	
	queue.finish()

	#cl.enqueue_copy(queue, h_numb_seq, d_numb_seq)
	
	#print(h_numb_seq[:textLen])
	#print(Counter(h_numb_seq))

	freqTab(queue, globalsize, None, N, M, K, numbKmer, d_numb_seq)

	queue.finish()

	cl.enqueue_copy(queue, h_numb_seq, d_numb_seq)

	print("Counting Done")

	#print(h_numb_seq[:textLen])
	#print(h_numb_seq[:numbKmer+2+4])
	assert(h_numb_seq[0] == 0), "File contains unknown nucleotide characters"

	return h_numb_seq[2:numbKmer+2+4]

def processOrganChromDict(organChromDict, nK, outfile):
	lNucls = ["A", "C", "G", "T"]
	lKmer = list(product(range(4), repeat=nK))
	with open(outfile, 'w') as f:	
		seqNameList = organChromDict.keys()
		for seqName in seqNameList:
			chromNumbs = organChromDict[seqName]
			chromFileList = ["../data/{}.chroms/chr{}.fa".format(seqName, numb) for numb in chromNumbs]
			freqs = np.array([kMerCount(file, nK).astype(np.int64) for file in chromFileList])
			freqs = np.sum(freqs, axis=0)
			monoCnts = freqs[:4]
			monoFreqs = monoCnts / monoCnts.sum()
			polyCnts = freqs[4:4+4**nK]
			polyFreqs = polyCnts / polyCnts.sum()
			polyExps = [reduce(mul, map((lambda x: monoFreqs[x]), kmer), 1) for kmer in lKmer]
			flds = "{:16}{:>16}{:>16}{:>16}\n"
			mono_entries = "{:16}{:16d}{:16.5f}\n"
			entries = "{:16}{:16d}{:16.5f}{:16.5f}\n"
			f.write("{} chromosomes statistics\n".format(seqName))
			f.write(flds.format("K-mer", "Counts", "Frequencies", "Expectations"))
			for i in range(4):
				f.write(mono_entries.format(lNucls[i], monoCnts[i], monoFreqs[i]))
			for i in range(4**nK):
				kmer = lKmer[i]
				kmer = reduce((lambda x,y: x+lNucls[y]), kmer, "")
				f.write(entries.format(kmer, polyCnts[i], polyFreqs[i], polyExps[i]))
			f.write("\n")

def main():
	dictChromCat = {'ce10': ['I', 'II', 'III', 'IV', 'V', 'X'], 'hg38': ([str(i) for i in range(1,23)]+['X','Y']), 'galGal3': ([str(i) for i in range(1,29)]+['32','W','Z']),
					'dm3': ['2R', '2L', '3R', '3L', '4', 'X']}
	K = 2
	outFile = "sandbox.txt"
	processOrganChromDict(dictChromCat, K, outFile)
	

if __name__=="__main__":
	rtime = time()
	main()
	rtime = time() - rtime
	print(rtime)