#ifndef FOURIER_TRANSFORM_H
#define FOURIER_TRANSFORM_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <math.h>

class FourierTransform
{
public:
    
    // computes 1d fft, assumes n is power of 2
    static bool fft1D(Eigen::VectorXd& data, const int isign);
    
    // swaps the left and right halves of the input frequency data
    static void fftShift1D(Eigen::VectorXd& shiftedfft, const Eigen::VectorXd& data);
    
    // computes magnitude of freqeuncy components
    static double abs1D(Eigen::VectorXd& data);
    
    // computes 2d fft, assumes m and n are powers of 2
    static bool fft2D(Eigen::MatrixXd& data, const int isign);
    
    // swaps the left and right halves of the input frequency data
    static void fftShift2D(Eigen::MatrixXd& shiftedfft, const Eigen::MatrixXd& data);
    
    // computes magnitude of freqeuncy components
    static double abs2D(Eigen::MatrixXd& data);
};

#endif
