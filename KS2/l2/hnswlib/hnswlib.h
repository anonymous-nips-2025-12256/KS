#pragma once
#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#else
#include <x86intrin.h>
#endif

#if defined(USE_AVX512)
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif
#endif

#include <queue>
#include <vector>
#include <iostream>
#include <string.h>
//#include <boost/random.hpp>
//#include <boost/random/normal_distribution.hpp>

namespace hnswlib {
    typedef size_t labeltype;

    template <typename T>
    class pairGreater {
    public:
        bool operator()(const T& p1, const T& p2) {
            return p1.first > p2.first;
        }
    };

    template<typename T>
    static void writeBinaryPOD(std::ostream &out, const T &podRef) {
        out.write((char *) &podRef, sizeof(T));
    }

    template<typename T>
    static void readBinaryPOD(std::istream &in, T &podRef) {
        in.read((char *) &podRef, sizeof(T));
    }

    template<typename MTYPE>
    using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);


    template<typename MTYPE>
    class SpaceInterface {
    public:
        //virtual void search(void *);
        virtual size_t get_data_size() = 0;

        virtual DISTFUNC<MTYPE> get_dist_func() = 0;

        virtual void *get_dist_func_param() = 0;

        virtual ~SpaceInterface() {}
    };

    template<typename dist_t>
    class AlgorithmInterface {
    public:
        virtual void addPoint(const void *datapoint, labeltype label, float* norm)=0;
		virtual void Boost(int* new_edge, int* new_size)=0;
        virtual void Add_Edge(int* new_edge, int new_size)=0;		

        virtual void RewriteSize(int id)=0;
		virtual void AddNewEdge(void* vec, int id, int max_M)=0;
		virtual void setEfc(int) = 0;
		
		virtual void addProjVal(int id, float*** LSH_vec, float* tmp_norm2, float* tmp_adjust, float* tmp_res, float*, int vecdim_, float* max_norm2, float* max_adjust, float* max_res,float*, unsigned short int* norm_quan, float* val, bool* is_zero, bool* is_edge)=0;
		virtual void addEdgeNorm(int id, float* tmp_norm2, float*, float*, float*, float diff2, float, float, float, bool*)=0;
        virtual void CalcEdgeNorm(int id, int vecdim, double* edge_norm, size_t* count);
		virtual void CalcEdgeVariance(double* norm2, int vecdim_, int data_size, double* avg);
        virtual void PermuteVec(int id, float** vec, int vecdim);
        virtual void Calc_wres(size_t edge_count, int vecdim0, int vecdim_, int data_size, int level);		
        virtual void find_neighbors(size_t vecdim_, float** train_org, int ind, int* count, float** data);		
		
		virtual void compression(int vecsize, int vecdim,  bool* is_zero)=0;
        virtual void query_rot(int dim_, int qsize, float* massQ, float* R)=0;		
        virtual void searchKnn(float *query_data, size_t k, unsigned int* result, float*** lsh_vec, float** query_lsh)const = 0;

        // Return k nearest neighbor in the order of closer fist
        virtual std::vector<std::pair<dist_t, labeltype>>
            searchKnnCloserFirst(const void* query_data, size_t k) const;

        virtual void saveIndex(const std::string &location, float diff2)=0;
        virtual ~AlgorithmInterface(){
        }
    };

    template<typename dist_t>
    std::vector<std::pair<dist_t, labeltype>>
    AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void* query_data, size_t k) const {
        std::vector<std::pair<dist_t, labeltype>> result;

        // here searchKnn returns the result in the order of further first
		
		/*
        auto ret = searchKnn(query_data, k);
        {
            size_t sz = ret.size();
            result.resize(sz);
            while (!ret.empty()) {
                result[--sz] = ret.top();
                ret.pop();
            }
        }
        */

        return result;
    }

}

#include "space_l2.h"
#include "space_ip.h"
#include "bruteforce.h"
#include "hnswalg.h"
