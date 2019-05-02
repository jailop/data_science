/* Class to manage dataset and operations.
 * Operations are designed to make classifitions
 * using the K-nearest neighbors algorithm
 *
 * (c) 2019 Jaime Lopez <jailop AT protonmail DOT com>
 */

#ifndef _DATASET_H
#define _DATASET_H

#include <armadillo>

using namespace arma;

class Dataset
{
    public:
        Dataset(uword n_entities, uword n_attributes);
        Dataset(uword n_entities, uword n_attributes, const char *csvfilename); 
        void normalize_linear();
        void normalize_stat();
        void make_train_test_indexes(double prop=0.8);
        double euclidean_distance(uword idx1, uword idx2);
        double predict_one(uword idx, int k=5);
        void predict(int k=5);
        uword n_entities;
        uword n_attributes;
        mat data;
        mat train, test;
        uvec train_index, test_index;
        vec y, y_hat;
};

#endif // _DATASET_H
