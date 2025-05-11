# ------------------------------------------------------------------------------
#  Parameters
# ------------------------------------------------------------------------------
name=glove
n=1183514  #data size
d=200      #dimension
qn=10000    #query size
k=10      #topk

efc=2000   #HNSW parameter
M=32       #HNSW parameter
L=10       #level(KS2 parameter)

dPath=./${name}/${name}_base.fvecs   #data path
qPath=./${name}/${name}_query.fvecs  #query path
tPath=./${name}/${name}_truth.ivecs        #groundtruth path

#----Indexing for the first execution and searching for the following executions---------

./build/KS2 ${dPath} ${qPath} ${tPath} ${n} ${qn} ${d} ${k} ${efc} ${M} ${L}

