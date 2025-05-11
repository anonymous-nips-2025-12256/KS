# ------------------------------------------------------------------------------
#  Parameters
# ------------------------------------------------------------------------------
name=glove
n=1183514  #data size
d=200      #dimension
qn=10000    #query size
k=10      #topk 
alg=1      #0:CEOs; 1:KS1(sym); 2:KS1(pol)  

dPath=./${name}/${name}_base.fvecs   #data path
qPath=./${name}/${name}_query.fvecs  #query path
tPath=./${name}/${name}_truth.ivecs        #groundtruth path

./build/KS1 ${dPath} ${qPath} ${tPath} ${n} ${qn} ${d} ${k} ${alg}

