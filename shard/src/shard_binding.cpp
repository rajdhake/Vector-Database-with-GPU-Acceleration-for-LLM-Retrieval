#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <stdexcept>

#include "flat_index.hpp"

namespace py = pybind11;

// CUDA hooks (weak-link if no CUDA build)
extern "C" void cosine_scores_kernel(const float*, const float*, float*, int, int);
extern "C" void l2_scores_kernel(const float*, const float*, float*, int, int);

#ifdef __CUDACC__
#include <cuda_runtime.h>
static void cuda_check(cudaError_t e){ if (e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
#endif

class FlatIndexGPU {
public:
    explicit FlatIndexGPU(int dim): dim_(dim), n_(0) {}
    int dim() const { return dim_; }
    int count() const { return n_; }

    int add_batch(py::array_t<float, py::array::c_style | py::array::forcecast> arr){
        if (arr.ndim()!=2) throw std::runtime_error("add_batch expects 2D array");
        auto N = (int)arr.shape(0); int D=(int)arr.shape(1);
        if (D!=dim_) throw std::runtime_error("dim mismatch");
        auto buf = arr.unchecked<2>();

        // append to host store (row-major) with normalization for cosine
        size_t old = xs_.size(); xs_.resize(old + (size_t)N*dim_);
        for(int i=0;i<N;++i){
            float norm=0.f; for(int d=0;d<dim_;++d){ float v=buf(i,d); xs_[old + (size_t)i*dim_ + d]=v; norm+=v*v; }
            norm = std::sqrt(std::max(norm,1e-12f));
            for(int d=0; d<dim_; ++d){ xs_[old + (size_t)i*dim_ + d] /= norm; }
        }
        n_ += N;
        // re-upload full matrix for simplicity (MVP)
#ifdef __CUDACC__
        if (!d_X_) cuda_check(cudaMalloc(&d_X_, (size_t)n_*dim_*sizeof(float)));
        else{
            float* new_ptr; cuda_check(cudaMalloc(&new_ptr, (size_t)n_*dim_*sizeof(float)));
            if (n_-N>0) cuda_check(cudaMemcpy(new_ptr, d_X_, (size_t)(n_-N)*dim_*sizeof(float), cudaMemcpyDeviceToDevice));
            cuda_check(cudaFree(d_X_)); d_X_=new_ptr;
        }
        cuda_check(cudaMemcpy(d_X_, xs_.data(), (size_t)n_*dim_*sizeof(float), cudaMemcpyHostToDevice));
#endif
        return n_-N; // starting index
    }

    py::list search(py::array_t<float, py::array::c_style | py::array::forcecast> q, int k, const std::string& metric){
        if (q.ndim()!=1 || (int)q.shape(0)!=dim_) throw std::runtime_error("query dim mismatch");
        if (n_==0) return py::list();

        std::vector<float> scores(n_);
#ifdef __CUDACC__
        // GPU path
        float *d_q=nullptr, *d_out=nullptr;
        cuda_check(cudaMalloc(&d_q, dim_*sizeof(float)));
        cuda_check(cudaMalloc(&d_out, n_*sizeof(float)));
        // If metric == cosine, normalize q on host first
        std::vector<float> qn(dim_);
        auto qb = q.unchecked<1>();
        float norm=0.f; for(int d=0; d<dim_; ++d){ qn[d]=qb(d); norm+=qn[d]*qn[d]; }
        norm = std::sqrt(std::max(norm,1e-12f)); for(int d=0; d<dim_; ++d) qn[d]/=norm;
        cuda_check(cudaMemcpy(d_q, qn.data(), dim_*sizeof(float), cudaMemcpyHostToDevice));
        dim3 grid(n_), block(256);
        if (metric == "cosine") cosine_scores_kernel<<<grid, block>>>(d_X_, d_q, d_out, n_, dim_);
        else l2_scores_kernel<<<grid, block>>>(d_X_, d_q, d_out, n_, dim_);
        cuda_check(cudaDeviceSynchronize());
        cuda_check(cudaMemcpy(scores.data(), d_out, n_*sizeof(float), cudaMemcpyDeviceToHost));
        cuda_check(cudaFree(d_q)); cuda_check(cudaFree(d_out));
#else
        // CPU fallback using FlatIndexCPU for simplicity
        if (!cpu_) cpu_.emplace(dim_);
        if ((int)cpu_->count() != n_) {
            // sync CPU mirror
            cpu_.emplace(dim_);
            cpu_->add_batch(xs_.data(), n_);
        }
        if (metric == "cosine"){
            auto res = cpu_->search_cosine(q.data(), k);
            py::list out; for (auto &r: res){ py::dict d; d["index"]=r.index; d["score"]=r.score; out.append(d);} return out;
        } else {
            auto res = cpu_->search_l2(q.data(), k);
            py::list out; for (auto &r: res){ py::dict d; d["index"]=r.index; d["score"]=r.score; out.append(d);} return out;
        }
#endif
        // top-k on CPU (partial sort)
        std::vector<int> idx(n_); for(int i=0;i<n_;++i) idx[i]=i;
        std::partial_sort(idx.begin(), idx.begin()+std::min(k,n_), idx.end(), [&](int a,int b){return scores[a]>scores[b];});
        py::list out;
        for (int i=0; i<std::min(k,n_); ++i){ int id = idx[i]; py::dict d; d["index"]=id; d["score"]=scores[id]; out.append(d);}
        return out;
    }

private:
    int dim_{}; int n_{};
    std::vector<float> xs_; // host storage (normalized rows)
#ifdef __CUDACC__
    float* d_X_ = nullptr; // device matrix (N x D)
#endif
    std::optional<FlatIndexCPU> cpu_{}; // fallback mirror
};

PYBIND11_MODULE(vdb, m){
    py::class_<FlatIndexGPU>(m, "FlatIndex")
        .def(py::init<int>())
        .def("dim", &FlatIndexGPU::dim)
        .def("count", &FlatIndexGPU::count)
        .def("add_batch", &FlatIndexGPU::add_batch)
        .def("search", &FlatIndexGPU::search, py::arg("q"), py::arg("k"), py::arg("metric") = std::string("cosine"));
}