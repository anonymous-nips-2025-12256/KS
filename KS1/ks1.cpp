#include <iostream>
#include <fstream>
#include <queue>
#include <cmath>
#include <chrono>
#include <time.h>
#include <omp.h>
#include <algorithm>
#include <unordered_set>

using namespace std;

struct Data{
    int id;
    float val;
};

bool compare(const Data &a, const Data &b) {
    return a.val > b.val;
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

void KS1(int data_size_, int query_size_, int dim_, char* path_q_, char* path_data_, char* truth_data_, int topk_, int alg) {
    int maxk = 100;
    size_t vecsize = data_size_;
    size_t qsize = query_size_;
    size_t vecdim = dim_;
    char path_index[1024];
    char *path_q = path_q_;
    char *path_data = path_data_;
    sprintf(path_index, "index.bin");

    int proj_num = 2048;
    float** LSH_vec = new float* [proj_num];
	for(int i = 0; i < proj_num; i++)
		LSH_vec[i] = new float[vecdim];
	
    float *massb = new float[vecdim];

    //cout << "Loading GT:\n";
    ifstream inputGT(truth_data_, ios::binary);
    unsigned int *massQA = new unsigned int[qsize * maxk];
    for (int i = 0; i < qsize; i++) {
        int t;
        inputGT.read((char *) &t, 4);
        inputGT.read((char *) (massQA + 100 * i), 4 * maxk);
    }
    inputGT.close();
	
    //cout << "Loading queries:\n";
    float *massQ = new float[qsize * vecdim];
    ifstream inputQ(path_q, ios::binary);

    for (int i = 0; i < qsize; i++) {
        int in = 0;
        inputQ.read((char *) &in, 4);
        inputQ.read((char *) massb, 4 * vecdim);
        float ssum = 0;
		for(int j = 0; j < vecdim; j++){
			ssum += massb[j] * massb[j]; 
		}
		ssum = sqrt(ssum);
        for (int j = 0; j < vecdim; j++) {
            massQ[i*vecdim+j] = massb[j] / ssum;
        }
    }
    inputQ.close();

    float *mass = new float[vecdim];
    ifstream input(path_data, ios::binary);
    int in = 0;

    //cout << "Building KS1 index:\n";

    float** vec = new float*[vecsize];
	for(int i = 0; i < vecsize; i++)
		vec[i] = new float[vecdim];

    for (int i = 0; i < vecsize; i++) {        
        input.read((char *) &in, 4);
        input.read((char *) vec[i], 4 * vecdim);	
	}	
	
    int half_num = proj_num / 2;

    int s0_arr[] = {2, 5, 10};
    std::vector<int> correct_num1(3,0);
	std::vector<int> correct_num2(3,0);
	std::vector<int> correct_num3(3,0);
	std::vector<int> correct_num4(3,0);

    int num_runs = 10;
    int tol_num = qsize * topk_ * num_runs;
    const char* filename = "all_batches.bin";
	std::ifstream in_proj(filename, std::ios::binary);	
	
    for(int round0 = 0; round0 < num_runs; round0++){
  	    for(int j = 0; j < half_num; j++){	
            float ssum = 0;		
  		    for(int l = 0; l < vecdim; l++){
  			    LSH_vec[j][l]= gaussian(0.0f, 1.0f);
			    ssum += (LSH_vec[j][l] * LSH_vec[j][l]);
  		    }
		      
		    ssum = sqrt(ssum);			
			if(alg == 1){		
  		        for(int l = 0; l < vecdim; l++){
                    LSH_vec[j][l]= LSH_vec[j][l] / ssum ;
  		        }        	
			}
			
  		    for(int l = 0; l < vecdim; l++){
  			    LSH_vec[j+half_num][l]= -1.0f*LSH_vec[j][l];
  		    }		
  	    }	
    
        if(alg == 2){
            for (int i = 0; i < proj_num; ++i) {
                in_proj.read(reinterpret_cast<char*>(LSH_vec[i]), vecdim * sizeof(float));
            }
		}
	
	    int scanned_k1 = 10;
	    int scanned_k2 = 100;
	    int scanned_k3 = 1000;
        int scanned_k4 = 10000;	
	    std::vector<std::vector<Data>> arr(proj_num, std::vector<Data>(vecsize));
    
	    printf("Round %d: building KS1 index...\n", round0+1);
#pragma omp parallel for	
        for(int i = 0; i < proj_num; i++){
	        for(int j = 0; j < vecsize; j++){
			    float sum = 0;
		        for(int l = 0; l < vecdim; l++){
				    sum += LSH_vec[i][l] * vec[j][l];
			    } 
                arr[i][j].id = j;
                arr[i][j].val = sum;			
	        }
		    sort(arr[i].begin(), arr[i].end(), compare);
        }

	    printf("Round %d: KS1 construction completed\n", round0+1);
	
        float tol_ip = 0;	
	    for(int i = 0; i < qsize; i++){
		    int max_id = 0;
	        float max_ip;
		    float* curQ = massQ + (i * vecdim);
		    std::vector<Data> arr2(proj_num);
	        for(int j = 0; j < proj_num; j++){
		        float cur_ip = 0;
		        for(int l = 0; l < vecdim; l++){
			        cur_ip += curQ[l] * LSH_vec[j][l];
		        }
			    if(j == 0){
				    arr2[j].id = 0;
				    arr2[j].val = cur_ip;
			    }
			    else{
				    arr2[j].id = j;
				    arr2[j].val = cur_ip;				
			    }
	        }
		    sort(arr2.begin(), arr2.end(), compare);	

            for(int s0 = 0; s0 < 3; s0++){		
		        std::vector<int> vec1(vecsize, 0);
		        std::vector<int> vec2(vecsize, 0);
		        std::vector<int> vec3(vecsize, 0);
		        std::vector<int> vec4(vecsize, 0);
		
                for(int round = 0; round < s0_arr[s0]; round++){
		            for(int j = 0; j < scanned_k1; j++){
			            for(int l = 0; l < topk_; l++){
				            int test_id = arr[arr2[round].id][j].id;
				            if(test_id == massQA[i*100+l] && vec1[test_id] == 0){
					            vec1[test_id] = 1;
					            correct_num1[s0]++;
					            break;
				            }
			            }
		            }

		            for(int j = 0; j < scanned_k2; j++){
			            for(int l = 0; l < topk_; l++){
				            int test_id = arr[arr2[round].id][j].id;
				            if(test_id == massQA[i*100+l] && vec2[test_id] == 0){
                                vec2[test_id] = 1;
					            correct_num2[s0]++;
					            break;
				            }
			            }
		            }
		
		            for(int j = 0; j < scanned_k3; j++){
			            for(int l = 0; l < topk_; l++){
				            int test_id = arr[arr2[round].id][j].id;
				            if(test_id == massQA[i*100+l] && vec3[test_id] == 0){
                                vec3[test_id] = 1;
					            correct_num3[s0]++;
					            break;
				            }
			            }
		            }

		            for(int j = 0; j < scanned_k4; j++){
			            for(int l = 0; l < topk_; l++){
				            int test_id = arr[arr2[round].id][j].id;
				            if(test_id == massQA[i*100+l] && vec4[test_id] == 0){
                                vec4[test_id] = 1;
					            correct_num4[s0]++;
					            break;
				            }
			            }
		            }			
		        }
		    }
	    }
	}	
	in_proj.close();
	for(int i = 0; i < 3; i++){
    printf("probe@%d: recall@10 = %f, recall@100 = %f, recall@1000 = %f, recall@10000 = %f\n", s0_arr[i], 1.0f*correct_num1[i]/tol_num, 1.0f*correct_num2[i]/tol_num, 1.0f*correct_num3[i]/tol_num, 1.0f*correct_num4[i]/tol_num);
    }
	return;    
}
