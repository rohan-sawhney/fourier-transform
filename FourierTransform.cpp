#include "FourierTransform.h"

bool powerOf2(const int n)
{
    if (n < 2 || n & (n-1)) {
        std::cerr << "n must be power of 2" << std::endl;
        return false;
    }
    
    return true;
}

// Source: Numerical Recipes Ch 12.2
bool FourierTransform::fft1D(Eigen::VectorXd& data, const int isign)
{
    const int n = (int)data.size() / 2;
    if (!powerOf2(n)) return false;
    
    int nn, mmax, m, j, istep, i;
    double wtemp, wr, wpr, wpi, wi, theta, tempr, tempi;
    
    // reverse binary indexing
    nn = n << 1;
    j = 1;
    for (i = 1; i < nn; i += 2) {
        if (j > i) {
            std::swap(data[j-1], data[i-1]);
            std::swap(data[j], data[i]);
        }
        
        m = n;
        while (m >= 2 && j > m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
    
    // Danielson-Lanczos
    mmax = 2;
    while (nn > mmax) {
        istep = mmax << 1;
        theta = isign * (2 * M_PI / mmax);
        wtemp = sin(0.5 * theta);
        wpr = -2.0 * wtemp * wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;
        for (m = 1; m < mmax; m += 2) {
            for (i = m; i <= nn; i += istep) {
                j = i + mmax;
                tempr = wr * data[j-1] - wi * data[j];
                tempi = wr * data[j] + wi * data[j-1];
                data[j-1] = data[i-1] - tempr;
                data[j] = data[i] - tempi;
                data[i-1] += tempr;
                data[i] += tempi;
            }
            wtemp = wr;
            wr = wr * wpr - wi * wpi + wr;
            wi = wi * wpr + wtemp * wpi + wi;
        }
        mmax = istep;
    }
    
    // scale inverse transform
    if (isign == -1) {
        for (i = 0; i < n; i++) {
            data[2*i] /= (double)n;
            data[2*i+1] /= (double)n;
        }
    }
    
    return true;
}

void FourierTransform::fftShift1D(Eigen::VectorXd& shiftedfft, const Eigen::VectorXd& data)
{
    const int n = (int)data.size() / 2;
    shiftedfft = data;
    
    int i;
    for (i = 0; i < n; i++) {
        std::swap(shiftedfft[i], shiftedfft[i+n]);
    }
}

double FourierTransform::abs1D(Eigen::VectorXd& data)
{
    const int n = (int)data.size() / 2;
    
    int i;
    double max = -INFINITY;
    for (i = 0; i < n; i++) {
        data[2*i] = sqrt(data[2*i]*data[2*i] + data[2*i+1]*data[2*i+1]);
        data[2*i+1] = 0.0;
        
        if (max < data[2*i]) max = data[2*i];
    }
    
    return max;
}

bool FourierTransform::fft2D(Eigen::MatrixXd& data, const int isign)
{
    const int nx = (int)data.cols() / 2;
    const int ny = (int)data.rows();
    
    if (!powerOf2(nx) || !powerOf2(ny)) return false;
    
    int i, j;
    
    // transform rows
    Eigen::VectorXd data1D(2*nx);
    for (i = 0; i < ny; i++) {
        for (j = 0; j < nx; j++) {
            data1D[2*j] = data(i, 2*j);
            data1D[2*j+1] = data(i, 2*j+1);
        }
        fft1D(data1D, isign);
        for (j = 0; j < nx; j++) {
            data(i, 2*j) = data1D[2*j];
            data(i, 2*j+1) = data1D[2*j+1];
        }
    }

    // transform col
    data1D.resize(2*ny);
    for (j = 0; j < nx; j++) {
        for (i = 0; i < ny; i++) {
            data1D[2*i] = data(i, 2*j);
            data1D[2*i+1] = data(i, 2*j+1);
        }
        fft1D(data1D, isign);
        for (i = 0; i < ny; i++) {
            data(i, 2*j) = data1D[2*i];
            data(i, 2*j+1) = data1D[2*i+1];
        }
    }
    
    return true;
}

void FourierTransform::fftShift2D(Eigen::MatrixXd& shiftedfft, const Eigen::MatrixXd& data)
{
    const int nx = (int)data.cols() / 2;
    const int ny = (int)data.rows() / 2;
    shiftedfft = data;
    
    int i, j;
    for (i = 0; i < ny; i++) {
        for (j = 0; j < nx; j++) {
            std::swap(shiftedfft(i, j), shiftedfft(i+ny, j+nx));
            std::swap(shiftedfft(i+ny, j), shiftedfft(i, j+nx));
        }
    }
}

double FourierTransform::abs2D(Eigen::MatrixXd& data)
{
    const int nx = (int)data.cols() / 2;
    const int ny = (int)data.rows();
    
    int i, j;
    double max = -INFINITY;
    for (i = 0; i < ny; i++) {
        for (j = 0; j < nx; j++) {
            data(i, 2*j) = sqrt(data(i, 2*j)*data(i, 2*j) + data(i, 2*j+1)*data(i, 2*j+1));
            data(i, 2*j+1) = 0.0;
            
            if (max < data(i, 2*j)) max = data(i, 2*j);
        }
    }
    
    return max;
}
