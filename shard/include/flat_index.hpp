#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cmath>

struct SearchResult { int index; float score; };

class FlatIndexCPU {
public:
    explicit FlatIndexCPU(int dim) : dim_(dim) {}
    int dim() const { return dim_; }

    // Adds n vectors (row-major). Returns starting index.
    int add_batch(const float* data, int n) {
        if (n <= 0) return count();
        size_t old = xs_.size();
        xs_.insert(xs_.end(), data, data + (size_t)n * dim_);
        // Pre-normalize for cosine (L2 norm = 1)
        for (int i = 0; i < n; ++i) {
            float norm = 0.f;
            for (int d = 0; d < dim_; ++d) norm += xs_[(old + (size_t)i*dim_) + d] * xs_[(old + (size_t)i*dim_) + d];
            norm = std::sqrt(std::max(norm, 1e-12f));
            for (int d = 0; d < dim_; ++d) xs_[(old + (size_t)i*dim_) + d] /= norm;
        }
        return (int)((old) / dim_);
    }

    int count() const { return (int)(xs_.size() / dim_); }

    std::vector<SearchResult> search_cosine(const float* q, int k) const {
        if (k <= 0) return {};
        std::vector<float> qn(q, q + dim_);
        // normalize q
        float norm=0.f; for (int d=0; d<dim_; ++d) norm+=qn[d]*qn[d];
        norm = std::sqrt(std::max(norm,1e-12f));
        for (int d=0; d<dim_; ++d) qn[d]/=norm;
        std::vector<SearchResult> heap; heap.reserve(k);
        auto push = [&](int idx, float score){
            if ((int)heap.size() < k) heap.push_back({idx, score});
            else {
                auto it = std::min_element(heap.begin(), heap.end(), [](auto&a,auto&b){return a.score<b.score;});
                if (score > it->score) *it = {idx, score};
            }
        };
        for (int i=0;i<count();++i){
            float acc=0.f; const float* x = &xs_[(size_t)i*dim_];
            for (int d=0; d<dim_; ++d) acc += x[d]*qn[d];
            push(i, acc);
        }
        std::sort(heap.begin(), heap.end(), [](auto&a,auto&b){return a.score>b.score;});
        return heap;
    }

    std::vector<SearchResult> search_l2(const float* q, int k) const {
        if (k <= 0) return {};
        std::vector<SearchResult> heap; heap.reserve(k);
        auto push = [&](int idx, float sim){
            if ((int)heap.size() < k) heap.push_back({idx, sim});
            else {
                auto it = std::min_element(heap.begin(), heap.end(), [](auto&a,auto&b){return a.score<b.score;});
                if (sim > it->score) *it = {idx, sim};
            }
        };
        for (int i=0;i<count();++i){
            const float* x = &xs_[(size_t)i*dim_];
            float dist=0.f; for (int d=0; d<dim_; ++d){ float diff=x[d]-q[d]; dist += diff*diff; }
            float sim = -dist; // turn into similarity (higher=better)
            push(i, sim);
        }
        std::sort(heap.begin(), heap.end(), [](auto&a,auto&b){return a.score>b.score;});
        return heap;
    }

private:
    int dim_;
    std::vector<float> xs_; // row-major
};