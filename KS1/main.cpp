#include <stdlib.h>

void KS1(int vecsize, int qsize, int dim, char* path_q_, char* path_data_, char* path_truth_, int topk, int alg);
int main(int argc, char** argv) {
    char* data_path = argv[1];
	char* query_path = argv[2];
	char* truth_path = argv[3];
	int vecsize = atoi(argv[4]);
	int qsize = atoi(argv[5]);
	int dim = atoi(argv[6]);
	int topk = atoi(argv[7]);	
	int alg = atoi(argv[8]);

    KS1(vecsize, qsize, dim, query_path, data_path, truth_path, topk, alg);
    return 0;
};
