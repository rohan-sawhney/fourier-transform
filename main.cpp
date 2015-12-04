#ifdef __APPLE_CC__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <Magick++.h>
#include "FourierTransform.h"

using namespace Magick;

int gridX = 512;
int gridY = 512;
int nx, ny;
int percent = 100;

bool fftData = false;
Eigen::MatrixXd data;
Eigen::MatrixXd magnitude;
Image image;

void printInstructions()
{
    std::cerr << "' ': reconstruct with specified frequency percentage\n"
              << "→/←: increase/decrease frequency percentage by 1%\n"
              << "↑/↓: increase/decrease frequency percentage by 5%\n"
              << "f: fft\n"
              << "i: ifft\n"
              << "r: reload image\n"
              << "escape: exit program\n"
              << std::endl;
}

void loadImage()
{
    fftData = false;
    
    image.read("/Users/rohansawhney/Desktop/developer/C++/fourier-transform/bunny.png");
    nx = (int)image.columns();
    ny = (int)image.rows();
    int range = pow(2, image.modulusDepth());
    PixelPacket *pixels = image.getPixels(0, 0, nx, ny);
    
    data.resize(ny, 2*nx);
    
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            Color color = pixels[ny * j + i];
            double c = color.blueQuantum() / range;
            if (c > 240) c = 0;
            
            data(i, 2*j) = c;
            data(i, 2*j+1) = 0.0;
        }
    }
}

void filterFrequencies()
{
    int low = ny * percent / 200;
    int high = ny - low;
    
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            if ((i > low && i < high) || (j > low && j < high)) {
                data(i, 2*j) = 0;
                data(i, 2*j+1) = 0;
            }
        }
    }
}

void normalizeMagnitude(const double max)
{
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            magnitude(i, 2*j) = log(magnitude(i, 2*j)) / log(max);
        }
    }
}

void init()
{
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0, gridX, gridY, 0.0);
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT);

    glBegin(GL_POINTS);
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            double c = fftData ? magnitude(i, 2*j) : data(i, 2*j) / 255.0;
            glColor4f(0.0f, 0.0f, c, 0.6f);
            glVertex2f(i, j);
        }
    }
    glEnd();
    
    glFlush();
}

void keyboard(unsigned char key, int x, int y)
{
    switch(key) {
        case 27 :
            exit(0);
        case ' ':
            loadImage();
            fftData = FourierTransform::fft2D(data, 1);
            if (fftData) {
                filterFrequencies();
                FourierTransform::fft2D(data, -1);
                fftData = false;
            }
            break;
        case 'f':
            if (!fftData) {
                fftData = FourierTransform::fft2D(data, 1);
                if (fftData) {
                    FourierTransform::fftShift2D(magnitude, data);
                    normalizeMagnitude(FourierTransform::abs2D(magnitude));
                }
            }
            break;
        case 'i':
            if (fftData) {
                FourierTransform::fft2D(data, -1);
                fftData = false;
            }
            break;
        case 'r':
            loadImage();
            break;
    }
    
    glutPostRedisplay();
}

void special(int i, int x0, int y0)
{
    switch (i) {
        case GLUT_KEY_UP:
            percent += 5;
            if (percent > 100) percent = 100;
            break;
        case GLUT_KEY_DOWN:
            percent -= 5;
            if (percent < 1) percent = 1;
            break;
        case GLUT_KEY_LEFT:
            percent -= 1;
            if (percent < 1) percent = 1;
            break;
        case GLUT_KEY_RIGHT:
            percent += 1;
            if (percent > 100) percent = 100;
            break;
    }
    
    std::stringstream title;
    title << "Fourier Transform, Frequency Percentage: " << percent << "%";
    glutSetWindowTitle(title.str().c_str());
    
    glutPostRedisplay();
}

int main(int argc, char** argv)
{
    printInstructions();
    InitializeMagick(*argv);
    loadImage();
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(gridX, gridY);
    std::stringstream title;
    title << "Fourier Transform, Frequency Percentage: " << percent << "%";
    glutCreateWindow(title.str().c_str());
    init();
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(special);
    glutMainLoop();
    
    return 0;
}
