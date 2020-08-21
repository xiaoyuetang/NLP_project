# NLP Final Project

### Part 1: 
```
$ cd nonCRF
$ python part1.py
```


### Part 2: 
```
$ cd nonCRF
$ python part2.py
```


### Part 4: 
```
$ cd CRF
```
Comment out all features except for 'U[0]:%s' % X[t][0]
```
$ python3 crf_train.py <path_to_train> modelfile_p4.json
$ python3 crf_test.py <path_to_dev.in>  modelfile_p4.json <path_to_dev.p4.out>
```


### Part 5 (ii):
```
$ cd CRF
```
Comment out all features except for 'U[0]:%s' % X[t][0] and 'POS_U[0]:%s' % X[t][1]
```
$  python3 crf_train.py <path_to_train> modelfile_p5ii.json
$  python3 crf_test.py <path_to_dev.in>  modelfile_p5ii.json <path_to_dev.p5.CRF.f4.out>
```


### Part 6 (i):
```
$  cd CRF
$  python3 crf_train.py <path_to_train> modelfile_p6i.json
$  python3 crf_test.py <path_to_dev.in>  modelfile_p6i.json <path_to_dev.p6.CRF.out>
```

### Evaluate: 
```
$ python evaluation.py
```
