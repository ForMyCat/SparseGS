import sys, os
from glob import glob
import io
f = io.open('./extractor_output.txt', 'r', encoding ='utf-8')
f2 = io.open('sparse/1/images.txt','r',  encoding ='utf-8')
f3 = io.open('sparse/1/images_reindexed.txt','w',  encoding ='utf-8')


counter = 1
id_dict = dict()
for i,l in enumerate(f.readlines()):
	words = l.split(' ')
	# print(words)
	if len(words) < 3:
		continue
	if words[2] == 'Name:':
		id_dict[words[-1]] = str(counter)
		print(counter, words[-1])
		counter += 1

print(id_dict)
	# name = (l[2:6])
    # if name == 'Name':

counter = 1
for i,l in enumerate(f2.readlines()):
	words = l.split(' ')
	if len(words) < 2:
		f3.write(l)
		continue
	f3.write(l.replace(str(counter), id_dict[words[-1]],1))
	counter += 1

f3.close()
f2.close()
f.close()

os.remove('sparse/1/images.txt')
os.replace('sparse/1/images_reindexed.txt', 'sparse/1/images.txt')




