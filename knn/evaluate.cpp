/* This program is aimed to evaluate performance
 * for K-nearest neighbors classification
 * based on optional dataset and controlling
 * kinds of normalization among other aspects
 *
 * (c) 2019 - Jaime Lopez <jailop AT protonmail DOT com>
 */

#include <iostream>
#include <getopt.h>
#include "dataset.h"

using namespace std;
using namespace arma;

int main(int argc, char **argv)
{
    char *filename = 0;  // Dataset filename
    int n_entities = 0, n_attributes = 0;  // Dataset dimensions
    double prop = 0.8;  // Proportion between train and test datasets
    int start = 1; // Starting number of neighbors
    int finish = 15;  // Finishing number of neighbors
    int normalize = 0; // Kind of normalization
    int c;

    /* Option setting */ 
    while ((c = getopt(argc, argv, "he:a:p:s:f:n:")) != -1) {
        switch (c) {
            case 'h':
                cout << "Usage:" << endl;
                cout << "  -e : Number of entities" << endl;
                cout << "  -a : Number of attributes" << endl;
                cout << "  -p : Proportion for train and test" << endl;
                cout << "  -s : Start for neighbors" << endl;
                cout << "  -f : Finish for neighbors" << endl;
                cout << "  -n : Normalize" << endl;
                cout << "       1 : Linear normatilization" << endl;
                cout << "       2 : Gaussian normailization" << endl;
                exit(0);
            case 'e':
                n_entities = atoi(optarg);
                break;
            case 'a':
                n_attributes = atoi(optarg);
                break;
            case 'p':
                prop = atol(optarg);
                break;
            case 's':
                start = atoi(optarg);
                break;
            case 'f':
                finish = atoi(optarg);
                break;
            case 'n':
                normalize = atoi(optarg);
        }
    }
    filename = argv[optind]; // Dataset filename

    /* Dataset configuration */
    Dataset data(n_entities, n_attributes, filename);
    if (normalize) {
        if (normalize == 1) // Linear
            data.normalize_linear();
        else if (normalize == 2)  // Gaussian
            data.normalize_stat();
    }
    // Generating training and testing datasets
    data.make_train_test_indexes(prop);

    /* Performance analysis */
    uword n = data.y.n_elem; // Number of elemenets of Y
    for (int i = start; i <= finish; i++) {
        int positive = 0;  // Counter for succesful predictions
        data.predict(i);
        for (uword j = 0; j < n; j++) 
            if (int(data.y_hat[j]) == int(data.y[j]))
                positive += 1;
        // Displaying accuracy
        cout << i << "," << positive * 1.0 / n << endl;
    }

    return 0;
}
