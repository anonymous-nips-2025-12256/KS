#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <time.h>
#include "hnswlib/hnswlib.h"
#include <unordered_set>

using namespace std;
using namespace hnswlib;

class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }

};

struct k_elem{
	int id;
	float dist;
};

int QsortComp(				// compare function for qsort
	const void* e1,						// 1st element
	const void* e2)						// 2nd element
{
	int ret = 0;
	k_elem* value1 = (k_elem *) e1;
	k_elem* value2 = (k_elem *) e2;
	if (value1->dist < value2->dist) {
		ret = -1;
	} else if (value1->dist > value2->dist) {
		ret = 1;
	} else {
		if (value1->id < value2->id) ret = -1;
		else if (value1->id > value2->id) ret = 1;
	}
	return ret;
}

int compare_int(const void *a, const void *b)
{
    return *(int*)a - *(int*)b;
}

static void
get_gt(unsigned int *massQA, float *massQ, float *mass, size_t vecsize, size_t qsize, L2Space &l2space,
       size_t vecdim, vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k) {


    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    DISTFUNC<float> fstdistfunc_ = l2space.get_dist_func();
    //cout << qsize << "\n";
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[100 * i + j]);
        }
    }
}

static float
test_approx(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg,
            vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, 
			 float*** lsh_vec, size_t vecdim, 
			float* query_rot, float** query_lsh, int* permutation) {    
	size_t correct = 0;
    size_t total = 0;

    for (int i = 0; i < qsize; i++) {
        unsigned int* result = new unsigned int[k];
		float* tmp = massQ + vecdim * i;
    
        for(int j = 0; j < vecdim; j++){
			int x = permutation[j];
			query_rot[j] = tmp[x];
		}
        		
        appr_alg.searchKnn(query_rot, k, result, lsh_vec, query_lsh);	
		
        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        for(int j = 0; j < k; j++) {
            if (g.find(result[j]) != g.end()) {
                correct++;
            } 
        }    
		delete[] result;      
    }	
    return 1.0f * correct / total;
}

static void
test_vs_recall(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, float*** lsh_vec, float* query_rot, float** query_lsh, int* permutation) {
    vector<size_t> efs;// = { 10,10,10,10,10 };
    	

    for (int i = 10; i < 100; i += 10) {
        efs.push_back(i);
    }
 
    for (int i = 100; i < 5000; i += 10) {
        efs.push_back(i);
    }
	
    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        StopW stopw = StopW();

        float recall = test_approx(massQ, vecsize, qsize, appr_alg, answers, k, lsh_vec, vecdim, query_rot, query_lsh, permutation);
	
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;

        cout << ef << "\t" << recall << "\t" << 1e6 / time_us_per_query << " QPS\n";
        if (recall > 1.0) {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}

float uniform(						// r.v. from Uniform(min, max)
	float min,							// min value
	float max)							// max value
{
	if (min > max) {printf("input error\n"); exit(0);}

	float x = min + (max - min) * (float) rand() / (float) RAND_MAX;
	if (x < min || x > max) {printf("input error\n"); exit(0);}

	return x;
}

float gaussian(						// r.v. from Gaussian(mean, sigma)
	float mu,							// mean (location)
	float sigma)						// stanard deviation (scale > 0)
{
	float PI = 3.141592654F;
    float FLOATZERO = 1e-6F;
	
	if (sigma <= 0.0f) {printf("input error\n"); exit(0);}

	float u1, u2;
	do {
		u1 = uniform(0.0f, 1.0f);
	} while (u1 < FLOATZERO);
	u2 = uniform(0.0f, 1.0f);
	
	float x = mu + sigma * sqrt(-2.0f * log(u1)) * cos(2.0f * PI * u2);
	return x;
}

void KS2(int efc_, int M_, int data_size_, int query_size_, int dim_, char* path_q_, char* path_data_, char* truth_data_, int L_, int topk_) {

	int efConstruction = efc_;
	int M = M_;
    int maxk = 100;
    size_t vecsize = data_size_;
      
    size_t qsize = query_size_;
	
	size_t vecdim;
	size_t orgdim = dim_;
	
	int real_level = L_;
	int residue = orgdim % real_level;
	if(residue == 0){
		vecdim = orgdim;
	}
	else{
		vecdim = orgdim + (real_level-residue);
	}	
	
    char path_index[1024];
    char *path_q = path_q_;
    char *path_data = path_data_;
    sprintf(path_index, "index.bin");

    int m = 128;
	int level = real_level;
	int LSH_level = real_level;
	int vecdim0 = vecdim / level;
    int LSH_vecdim0 = vecdim / LSH_level;

    float*** LSH_vec = new float** [LSH_level];
	for(int i = 0; i < LSH_level; i++)
		LSH_vec[i] = new float* [m];

    for(int i = 0; i < LSH_level; i++){
	    for(int j = 0; j < m; j++){		
		    LSH_vec[i][j] = new float[LSH_vecdim0];
	    }	
    } 
	
    int train_size = 100000;	
	float min_norm0, max_norm0, diff0;
  
    float *massb = new float[vecdim];

    ifstream inputGT(truth_data_, ios::binary);
    unsigned int *massQA = new unsigned int[qsize * maxk];
    for (int i = 0; i < qsize; i++) {
        int t;
        inputGT.read((char *) &t, 4);
        inputGT.read((char *) (massQA + 100 * i), 4 * maxk);
    }
    inputGT.close();
	
    float *massQ = new float[qsize * vecdim];
    ifstream inputQ(path_q, ios::binary);

    for (int i = 0; i < qsize; i++) {
        int in = 0;
        inputQ.read((char *) &in, 4);
        inputQ.read((char *) massb, 4 * orgdim);
		
		float norm_sum = 0;
		for(int j = 0; j < vecdim; j++){
			norm_sum += massb[j] * massb[j];	
	    }
		norm_sum = sqrt(norm_sum);
		for(int j = 0; j < vecdim; j++){
			massb[j] = massb[j] / norm_sum;	
		}		
		
        for (int j = 0; j < orgdim; j++) {
            massQ[i * vecdim + j] = massb[j];
        }
        for (int j = orgdim; j < vecdim; j++) {
            massQ[i * vecdim + j] = 0;
        }
    }
    inputQ.close();

    float *mass = new float[vecdim];
    ifstream input(path_data, ios::binary);
    int in = 0;
    L2Space l2space(vecdim);
	InnerProductSpace ipsubspace(vecdim0);
	InnerProductSpace LSHsubspace(LSH_vecdim0);
	InnerProductSpace ipspace(vecdim);  //new
    int* permutation = new int[vecdim];	

	float read_diff2;	

    HierarchicalNSW<float> *appr_alg;
    if (exists_test(path_index)) {
        cout << "Loading index from " << path_index << ":\n";
			
		ifstream input2("ProjInfo", ios::binary);

		input2.read((char*)(&read_diff2), sizeof(float));	
		
	    for(int j = 0; j < LSH_level; j++){
		    for(int i = 0; i < m; i++){
                input2.read((char*) LSH_vec[j][i] , sizeof(float) * LSH_vecdim0);
		    }	
		}		
		
		input2.read((char*)(permutation), sizeof(int) * vecdim);
	    
        appr_alg = new HierarchicalNSW<float>(&l2space, &ipsubspace, &ipspace, &LSHsubspace, path_index, read_diff2, LSH_level, LSH_vecdim0, false);
    } else {
        cout << "Building HNSW index...\n";	

        float** vec = new float*[vecsize];
	    for(int i = 0; i < vecsize; i++)
		    vec[i] = new float[vecdim];

        for (int i = 0; i < vecsize; i++) {        
            input.read((char *) &in, 4);

            input.read((char *) vec[i], 4 * orgdim);
			
			for(int j = orgdim; j < vecdim; j++)
				vec[i][j] = 0;
			
			float norm_sum = 0;
			for(int j = 0; j < vecdim; j++){
			    norm_sum += vec[i][j] * vec[i][j];	
			}
			norm_sum = sqrt(norm_sum);
			for(int j = 0; j < vecdim; j++){
			    vec[i][j] = vec[i][j] / norm_sum;	
			}
			
	    }

        unsigned short int* norm_quan = new unsigned short int[vecsize];									
        appr_alg = new HierarchicalNSW<float>(m, diff0, level, vecdim0, LSH_level, LSH_vecdim0, &l2space, &ipsubspace, &ipspace, &LSHsubspace, vecsize, M, efConstruction);
	
  	    for(int i = 0; i < LSH_level; i++){ 
  	        for(int j = 0; j < m; j++){	
                float ssum = 0;		
  		        for(int l = 0; l < LSH_vecdim0; l++){
  			        LSH_vec[i][j][l]= gaussian(0.0f, 1.0f);
				    ssum += (LSH_vec[i][j][l] * LSH_vec[i][j][l]);
  		        }
			    ssum = sqrt(ssum);
			    float new_norm = sqrt(LSH_level); 
  		        for(int l = 0; l < LSH_vecdim0; l++){
                    LSH_vec[i][j][l]= LSH_vec[i][j][l] / ssum / new_norm;
  		        }			
  	        }
        }

        float* norm = new float[vecsize];
        appr_alg->addPoint((void *) vec[0], (size_t) 0, &(norm[0]));
	
        int j1 = 0;
        StopW stopw = StopW();
        StopW stopw_full = StopW();
        size_t report_every = 10000;

#pragma omp parallel for schedule(dynamic)
        for (int i = 1; i < vecsize; i++) {
            appr_alg->addPoint((void *) vec[i], (size_t) i, &(norm[i]));
        }
		cout << "HNSW Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";

        StopW stopw_ks2 = StopW();		
        int max_M = 100;
        appr_alg->setEfc(max_M);

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < vecsize; i++) {
            appr_alg->AddNewEdge((void*) (vec[i]), i, max_M);
        }
	
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < vecsize; i++) {
            appr_alg->RewriteSize(i);
        }	

        double* norm2 = new double[vecdim];

		k_elem* sort_arr = new k_elem[vecdim];
		k_elem* cur_sum = new k_elem[level];
		for(int i = 0; i < level; i++){
			cur_sum[i].id = i;
			cur_sum[i].dist = 0;
		}
		
		k_elem* temp = new k_elem[level];
		int** id_arr = new int*[level];
		for(int i = 0; i < level; i++)
			id_arr[i] = new int[vecdim0];

        size_t edge_count = 0;
		for(int i = 0; i < vecdim; i++) norm2[i] = 0;
		
		size_t* edge_size = new size_t[vecsize];
		for(int i = 0; i < vecsize; i++){
			edge_size[i] = 0;
		}
		double** edge_norm = new double*[vecsize];
		for(int i = 0; i < vecsize; i++){
			edge_norm[i] = new double[vecdim];
			for(int j = 0; j < vecdim; j++){
				edge_norm[i][j] = 0;
			}
		}
		
#pragma omp parallel for schedule(dynamic)
		for(int i = 0; i < vecsize; i++){ 
	        appr_alg->CalcEdgeNorm(i, vecdim, edge_norm[i], &(edge_size[i]));
		}	
        for(int i = 0; i < vecsize; i++)		
            edge_count += edge_size[i];

        for(int i = 0; i < vecsize; i++){
            for(int j = 0; j < vecdim; j++){		
                norm2[j] += edge_norm[i][j] / edge_count;	
			}				
	    }
		
		for(int i = 0; i < vecdim; i++){
			sort_arr[i].id = i;
			sort_arr[i].dist = (float)(norm2[i]);
		}

        qsort(sort_arr, vecdim, sizeof(k_elem), QsortComp);

        int half_dim = vecdim0 / 2;
		
		int t = 0;
        for(int i = 0; i < half_dim; i++){
			for(int j = 0; j < level; j++){
				temp[j].dist = sort_arr[i * level + j].dist + sort_arr[vecdim - 1 - (i * level + j)].dist;
			    temp[j].id = j;
			}
			
			qsort(temp, level, sizeof(k_elem), QsortComp);
		    for(int j = 0; j < level; j++){
			    cur_sum[j].dist = cur_sum[j].dist + temp[level - 1 - j].dist;
			    int l = cur_sum[j].id;
				int k = temp[level - 1 - j].id;
			    id_arr[l][t] = sort_arr[i * level + k].id;
				id_arr[l][t+1] =sort_arr[vecdim - 1 - (i * level + k)].id;
			}
			t += 2;
			qsort(cur_sum, level, sizeof(k_elem), QsortComp);
		}
		
		if(vecdim0 % 2 != 0){
			int i = vecdim0 / 2;
			for(int j = 0; j < level; j++){
				temp[j].dist = sort_arr[i * level + j].dist;
			    temp[j].id = j;
			}
			
			qsort(temp, level, sizeof(k_elem), QsortComp);
		    for(int j = 0; j < level; j++){
			    cur_sum[j].dist = cur_sum[j].dist + temp[level - 1 - j].dist;
			    int l = cur_sum[j].id;
				int k = temp[level - 1 - j].id;
			    id_arr[l][t] = sort_arr[i * level + k].id;
			}
			t += 1;			
		}

        for(int i = 0; i < level; i++){
			qsort(id_arr[i], vecdim0, sizeof(int), compare_int);
		}
		
		for(int i = 0; i < level; i++){
			for(int j = 0; j < vecdim0; j++){
				permutation[i * vecdim0 + j] = id_arr[i][j];
			}
		}
			
		float* tmp_arr = new float[vecdim];
		
		for(int i = 0; i < vecsize; i++){			
			float* tmp = vec[i];
		    for(int j = 0; j < vecdim; j++){
			    int x = permutation[j];
			    tmp_arr[j] = tmp[x];
		    }
		    for(int j = 0; j < vecdim; j++){
			    tmp[j] = tmp_arr[j];
		    }						
		}

		for(int i = 0; i < vecdim; i++){
			int count_ = 0;
			for(int j = 0; j < vecdim; j++){
				if(permutation[j] == i){
					count_++;
				}				
			}
			if(count_ != 1){
				printf("error count = %d, count_\n");
				exit(0);
			}	
		}

#pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < vecsize; i++)			
		    appr_alg->PermuteVec(i, vec, vecdim);

        for(int i = 0; i < vecsize; i++) {delete[] edge_norm[i];}
	
	    delete[] edge_norm;
	    delete[] edge_size;

		float** tmp_norm2 = new float*[vecsize];
		for(int i = 0; i < vecsize; i++)
			tmp_norm2[i] = new float[2 * M];
		
		float** tmp_adjust = new float*[vecsize];
		for(int i = 0; i < vecsize; i++)
			tmp_adjust[i] = new float[2 * M];

		float** tmp_res = new float*[vecsize];
		for(int i = 0; i < vecsize; i++)
			tmp_res[i] = new float[2 * M];	

		float** tmp_last = new float*[vecsize];
		for(int i = 0; i < vecsize; i++)
			tmp_last[i] = new float[2 * M];

		bool** is_edge = new bool*[vecsize];
		for(int i = 0; i < vecsize; i++)
			is_edge[i] = new bool[2 * M];			
		
		float* max_norm2 = new float[vecsize];
		float* max_adjust = new float[vecsize];
		float* max_res = new float[vecsize];
		float* max_last = new float[vecsize];
		
		bool* is_zero = new bool[vecsize];
		for(int i = 0; i < vecsize; i++) is_zero[i] = false;
		
		j1 = 0;
		float* test_val = new float[vecsize];
		for(int i = 0; i < vecsize; i++) test_val[i] = 0;	

#pragma omp parallel for schedule(dynamic)
	    for (int i = 0; i < vecsize; i++){
	        appr_alg->addProjVal(i, LSH_vec, tmp_norm2[i], tmp_adjust[i], tmp_res[i], tmp_last[i], vecdim, &(max_norm2[i]), &(max_adjust[i]), &(max_res[i]), &(max_last[i]), norm_quan, &(test_val[i]), &(is_zero[i]), is_edge[i]);
	    }

		float tol_max_norm2 = max_norm2[0];
        float tol_max_adjust = max_adjust[0]; 
		float tol_max_res = max_res[0];	
        float tol_max_last = max_last[0];			
				
        for(int i = 1; i < vecsize; i++){
            if(max_norm2[i] > tol_max_norm2) {tol_max_norm2 = max_norm2[i];}
            if(max_adjust[i] > tol_max_adjust) {tol_max_adjust = max_adjust[i];}  
            if(max_res[i] > tol_max_res) {tol_max_res = max_res[i];}
            if(max_last[i] > tol_max_last) {tol_max_last = max_last[i];}					
		}

		delete[] max_norm2;

        int interval = 256*256;  
        int interval0 = 32767;

		float diff2 = (tol_max_norm2 - 0) / (interval0 - 1);  
		float diffadj = (tol_max_adjust - 0) / (interval - 1);  
		float diffres = (tol_max_res - 0) / (interval - 1);
		float difflast = (tol_max_last - 0) / (interval0 - 1);

	    for (int i = 0; i < vecsize; i++){
	        appr_alg->addEdgeNorm(i, tmp_norm2[i], tmp_adjust[i], tmp_res[i], tmp_last[i], diff2, diffadj, diffres, difflast, is_edge[i]);
	    }	

	    appr_alg->compression(vecsize, vecdim, is_zero);
	
        input.close();
        cout << "KS2 build time:" << 1e-6 * stopw_ks2.getElapsedTimeMicro() << "  seconds\n";
				
		read_diff2 = diff2;
		
        appr_alg->saveIndex(path_index, read_diff2);
		
		ofstream output("ProjInfo", ios::binary);
         
		output.write((char*)(&read_diff2), sizeof(float));
         
        for(int j = 0; j < LSH_level; j++){
		    for(int i = 0; i < m; i++){
	            output.write((char*)(LSH_vec[j][i]), sizeof(float) * LSH_vecdim0);
		    }
		}		
        
        for(int j = 0; j < level; j++){
	        output.write((char*)(id_arr[j]), sizeof(int) * vecdim0);
		}	
		
		output.close();
		printf("Indexing finished\n");
		exit(0);
    }

    vector<std::priority_queue<std::pair<float, labeltype >>> answers;
    size_t k = topk_;
    get_gt(massQA, massQ, mass, vecsize, qsize, l2space, vecdim, answers, k);

	float* query_rot = new float[vecdim];
	
	float** query_lsh = new float*[LSH_level];
	for(int i = 0; i < LSH_level; i++) query_lsh[i] = new float[2 * m];   	
	
    for (int i = 0; i < 1; i++)
        test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k, LSH_vec, query_rot, query_lsh, permutation);
    return;
}
