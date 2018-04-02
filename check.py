from collections import Counter

dictChromCat = {'ce10': ['I', 'II', 'III', 'IV', 'V', 'X'], 'hg38': ([str(i) for i in range(1,23)]+['X','Y']), 'galGal3': ([str(i) for i in range(1,29)]+['32','W','Z']),
					'dm3': ['2R', '2L', '3R', '3L', '4', 'X']}

seqName = list(dictChromCat.keys())[0]
chromNumbs = dictChromCat[seqName]

chromFileList = ["../data/{}.chroms/chr{}.fa".format(seqName, numb) for numb in chromNumbs]

data = "".encode('utf-8')
for file in chromFileList:
	f = open(file, 'rb')
	fileData = f.read().upper().splitlines()
	f.close()
	data += ("".encode('utf-8')).join(fileData)

print(Counter(data))