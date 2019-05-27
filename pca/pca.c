#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_complex_double.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_statistics_double.h>

#define NCOL 12
#define NROW 4898
#define DATAFILE "data/tmp/winequality-white.csv"

gsl_matrix *matrix_load(const char *filename)
{
    int res;
    FILE *fin;
    gsl_matrix *m;
    fin = fopen(DATAFILE, "r");
    m = gsl_matrix_alloc(NROW, NCOL);
    res = gsl_matrix_fscanf(fin, m);
    if (res == GSL_EFAILED) {
        fprintf(stderr, "Error: matrix cannot be read");
        exit(EXIT_FAILURE);
    }
    fclose(fin);
    return m; 
}

void matrix_write(gsl_matrix *m, const char *filename)
{
    int i, j;
    char c;
    FILE *fout;
    fout = fopen(filename, "w");
    if (!fout) {
        fprintf(stderr, "File to write matrix cannot be open\n");
        return;
    }
    for (i = 0; i < m->size1; i++) {
        for (j = 0; j < m->size2; j++) {
            if (j == m->size2 - 1)
                c = '\n';
            else
                c = '\t';
            fprintf(fout, "%lf%c", gsl_matrix_get(m, i, j), c);
        }
    }
    fclose(fout);
}

gsl_matrix *matrix_complex_real(gsl_matrix_complex *m)
{
    int j;
    gsl_matrix *ret;
    gsl_vector_complex_view col_complex;
    gsl_vector_view col;
    ret = gsl_matrix_alloc(m->size1, m->size2);
    for (j = 0; j < m->size2 - 1; j++) {
        col_complex = gsl_matrix_complex_column(m, j);
        col = gsl_vector_complex_real(&col_complex.vector);
        gsl_matrix_set_col(ret, j, &col.vector);
    }
    return ret;
}

gsl_matrix *matrix_covariance(gsl_matrix *m)
{
    gsl_matrix *cov;
    cov = gsl_matrix_alloc(m->size1, m->size1);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, m, m, 0.0, cov);
    gsl_matrix_scale(cov, (double) m->size1);
    return cov;
}

gsl_matrix *matrix_normalize_norm(gsl_matrix *m)
{
    int j;
    double mean, stddev;
    gsl_matrix *ret;
    gsl_vector_view col;
    ret = gsl_matrix_alloc(m->size1, m->size2);
    gsl_matrix_memcpy(ret, m);
    for (j = 0; j < ret->size2; j++) {
        col = gsl_matrix_column(ret, j);
        mean = gsl_stats_mean(col.vector.data, col.vector.size, col.vector.stride);
        stddev = gsl_stats_sd(col.vector.data, col.vector.size, col.vector.stride);
        gsl_vector_add_constant(&col.vector, -mean);
        gsl_vector_scale(&col.vector, 1/stddev);
    }
    return ret;
}

int main(int argc, char *argv[])
{
    gsl_matrix *data;
    gsl_matrix *cov;
    gsl_matrix *X;
    gsl_matrix_view aux;
    gsl_vector_view y;
    gsl_vector_complex *eval;
    gsl_matrix *evec_real;
    gsl_matrix_complex *evec;
    gsl_eigen_nonsymmv_workspace *w;
    gsl_matrix *X_pca;

    data = matrix_load(DATAFILE);
    
    aux = gsl_matrix_submatrix(data, 0, 0, NROW - 1, NCOL - 2);;
    X = matrix_normalize_norm(&aux.matrix);
    matrix_write(X, "data/tmp/data_normalized_x.csv");
    y = gsl_matrix_column(data, NCOL - 1);
    cov = matrix_covariance(X);
    matrix_write(cov, "data/tmp/covariance.csv");

    eval = gsl_vector_complex_alloc(cov->size1);
    evec = gsl_matrix_complex_alloc(cov->size1, cov->size1);
    w = gsl_eigen_nonsymmv_alloc(cov->size1);
    gsl_eigen_nonsymmv(cov, eval, evec, w);
    gsl_eigen_nonsymmv_free(w);

    X_pca = gsl_matrix_alloc(X->size1, X->size2);
    
    evec_real = matrix_complex_real(evec);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, evec_real, X, 0.0, X_pca);
    matrix_write(X_pca, "data/tmp/x_pca.csv");

    gsl_matrix_free(data);
    gsl_matrix_free(X);
    gsl_matrix_free(X_pca);
    gsl_matrix_free(cov);
    gsl_matrix_free(evec_real);
    gsl_matrix_complex_free(evec);
    gsl_vector_complex_free(eval);
    return 0;
}
