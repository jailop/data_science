#include <map>
#include "dataset.h"

using namespace std;

double euclidean_distance(double x1, double x2)
{
    double diff = x1 - x2;
    return sqrt(diff * diff);
}

Dataset::Dataset(uword n_entities, uword n_attributes)
    : n_entities(n_entities), n_attributes(n_attributes), data(n_entities, n_attributes)
{

}

Dataset::Dataset(uword n_entities, uword n_attributes, const char *csvfilename)
    : n_entities(n_entities), n_attributes(n_attributes), data(n_entities, n_attributes)
{
    data.load(csvfilename, csv_ascii);
}

void Dataset::normalize_linear() {
    for (uword i = 0; i < this->n_attributes - 1; i++) {
        vec col = this->data.col(i);
        double min = col.min();
        double max = col.max();
        for (uword j = 0; j < this->n_entities; j++) {
            this->data(j, i) -= max;
            this->data(j, i) /= min - max;
        }
    }
}

void Dataset::normalize_stat() {
    rowvec x = mean(this->data);
    rowvec s = stddev(this->data);
    for (uword i = 0; i < this->n_attributes - 1; i++) {
        for (uword j = 0; j < this->n_entities; j++) {
            this->data(j, i) -= x[i];
            this->data(j, i) /= s[i];
        }
    }
}

void Dataset::make_train_test_indexes(double prop)
{
    uvec idx(this->n_entities);
    for (uword i = 0; i < this->n_entities; i++)
        idx[i] = i;
    uvec shf(this->n_entities);
    shf = shuffle(idx);
    uword n = prop * this->n_entities;
    this->train_index = shf.head(n);
    this->test_index = shf.tail(this->n_entities - n);
    this->train = this->data.rows(this->train_index);
    this->test = this->data.rows(this->test_index);
    this->y = this->test.col(n_attributes - 1);
}

double Dataset::euclidean_distance(uword idx1, uword idx2)
{
    uword d = this->n_attributes - 1;
    auto x1 = this->data.row(idx1).subvec(0, d);
    auto x2 = this->data.row(idx2).subvec(0, d);
    auto diff = x1 - x2;
    return sqrt(sum((diff % diff)));
}

double Dataset::predict_one(uword idx, int k)
{
    uword n = this->train_index.n_elem;
    map<int, int> neighbors;
    vec distances(n);
    for (uword i = 0; i < n; i++)
        distances[i] = this->euclidean_distance(this->test_index[idx], this->train_index[i]);
    for (int i = 0; i < k; i++) {
        int min_pos = distances.index_min();
        int data_idx = this->train_index[min_pos];
        int label = (int) this->data.row(data_idx)[this->n_attributes - 1];
        neighbors[label] += 1;
        distances[min_pos] = NAN;
    }
    map<int, int>::iterator it = neighbors.begin();
    int max = 0;
    int label = it->first;
    while(it != neighbors.end()) {
        if (it->second > max) {
            label = it->first;
            max = it->second;
        }
        it++;
    }
    return label;
}

void Dataset::predict(int k)
{
    uword n = test_index.n_elem;
    this->y_hat = vec(n); 
    for (uword i = 0; i < n; i++) 
        this->y_hat[i] =  this->predict_one(i, k);
}
