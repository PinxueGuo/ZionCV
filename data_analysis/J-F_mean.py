import re

file_name = '/home/guopx/VOS/MAST-src/results/test_epoch19/benchmark.log'

Js = []
Fs = []

for line in open(file_name):
    match = re.findall(r'\d+.\d+', line)
    if match:
        Js.append( float(match[-2]) )
        Fs.append( float(match[-1]) )
JsMean = sum(Js)/len(Js)
FsMean = sum(Fs)/len(Fs)
print('Js:', JsMean)
print('Fs:', FsMean)
print((JsMean+FsMean)/2)
