# ------------------------------------------------------------------------------
#  Parameters
# ------------------------------------------------------------------------------
name=gist
n=1000000  #data size
d=960      #dimension
qn=1000    #query size
k=10      #topk

efc=1000   #HNSW parameter
M=32       #HNSW parameter
L=20       #level(KS2 parameter) 

dPath=./${name}/${name}_base.fvecs   #data path
qPath=./${name}/${name}_query.fvecs  #query path
tPath=./${name}/${name}_truth.ivecs        #groundtruth path

#----Indexing for the first execution and searching for the following executions---------

./build/KS2 ${dPath} ${qPath} ${tPath} ${n} ${qn} ${d} ${k} ${efc} ${M} ${L}

