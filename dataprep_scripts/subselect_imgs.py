import sys, os
from glob import glob
# assert os.path.isfile(sys.argv[1])
f = open(sys.argv[1], 'r')
imgfs = glob(sys.argv[2] + '\*')
imgfs = [os.path.basename(x) for x in imgfs]
print(imgfs)

f2 = open(sys.argv[1][:-4]+'2.txt', 'w')
for i,l in enumerate(f.readlines()):
	if i < 4:
		continue
	if i % 2 == 1:
		f2.write('\n')
	else:
		f2.write(l)

f.close()
f2.close()
f2 = open(sys.argv[1][:-4]+'2.txt', 'r')
f3 = open(sys.argv[1][:-4]+'3.txt', 'w')
lines = f2.readlines()

ctr = 1

for idx, line in enumerate(lines):
	splt = line.split(' ')
	if splt[-1].strip() in imgfs:
		splt[0] = str(ctr)
		print(' '.join(splt))
		f3.write(' '.join(splt))
		f3.write('\n')
		ctr += 1
	else:
		continue
	if (ctr == len(imgfs)+1):
		break
f2.close()
f3.close()
os.remove('sparse/1/images.txt')
os.replace('sparse/1/images3.txt', 'sparse/1/images.txt')
os.remove('sparse/1/images2.txt')
open('sparse/1/points3D.txt', 'w').close()



