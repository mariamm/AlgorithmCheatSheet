#pragma once

#include "Common.h"

//Find the equation of a line for a segment(a line segment is a part of a line that is bounded by two distinct end points)
/*
* Given segment PQ (Px,Py,Qx,Qy)
* Find a line in aplane Ax+By+C=0 which passes through the segment
* A=Py−Qy,
* B=Qx−Px,
* C=−APx−BPy.
*/



//Euclidean distance between two points in 2d (l2 norm, continues diagonal allowed)
// sqrt( (x2-x1)^2 + (y2-y1)^2)
double euclideanDistance(pair<int, int> p1, pair<int, int> p2)
{
    double deltax_squared = pow(p2.first - p1.first, 2);
    double deltay_squared = pow(p2.second - p1.second, 2);
    return sqrt(deltax_squared + deltay_squared);
}
//Manhatten distance between two points (l1 norm, stairs like)
// |x2-x1| + |y2-y1|
int manhattenDistance(pair<int, int> p1, pair<int, int> p2)
{
    return abs(p1.first - p2.first) + abs(p1.second - p2.second);
}

//Chebychev distance between two points (kings move, discrete diagonal allowed)
// max(|x2-x1|, |y2-y1|)
int chebychevDistance(pair<int, int> p1, pair<int, int> p2)
{
    return max(abs(p1.first - p2.first), abs(p1.second - p2.second));
}

//Dot product (Dot product of perpendicular vectors is 0)
int dotProduct(vector<int> v, vector<int> w)
{
    assert(v.size() == w.size());
    size_t size = v.size();
    int dot = 0;
    for (size_t i = 0; i < size; i++)
    {
        dot += (v[i] * w[i]);
    }
    return dot;
}

//norm (square length) of a vector can also be defined with dot product |a|^2 = a dot a
int norm(vector<int> a)
{
    return dotProduct(a, a);
}

double vectorLength(vector<int> a)
{
    return sqrt(norm(a));
}
//projection (also known as the vector component or vector resolution of a in the direction of b) â1 = a1b
//Intuition: length of the projection of a onto b. Dot product of perpendicular vectors is 0 
double projection(vector<int> a, vector<int> b)
{
    return dotProduct(a, b) / vectorLength(b);
}
double angle(vector<int> a, vector<int> b)
{
    return acos(dotProduct(a, b) / (vectorLength(a) * vectorLength(b)));
}
//Cross product
//Intuition: The area of the parallelogram formed by v and w represented as a vector perpendicular to v x w
//alternative calculation   : v x w = det([v_x w_x]) 
//                                        [v_y w_y] 
vector<int> crossProduct3(vector<int> v, vector<int> w)
{
    vector<int> cross(3);
    cross[0] = v[1] * w[2] - v[2] * w[1];
    cross[1] = v[2] * w[0] - v[0] * w[2];
    cross[2] = v[0] * w[1] - v[1] * w[0];
    return cross;
}

//there is no cross product in 2d, however setting the z component of the inputs to zero will result
//to a vector with 0 in the x,y components and return the scalar value of z 
//also called the perp dot product & equivalent to the determinant of a square matrix
int crossProduct2(vector<int> v, vector<int> w)
{
    return v[0] * w[1] - v[1] * w[0];
}
//Determinant of a matrix
//Intuition: how much the area/volume of 1 (unit vectors) is stretched by this matrix.
//Det = 0 -> the transformation causes a reduction to a smaller dimension (the columns of the matrix are lineraly dependent)
//Negative det : inverts the orientation of space
int determinant2(vector<vector<int>> A)
{
    assert(A.size() == 2);
    assert(A[0].size() == 2);
    return A[0][0] * A[1][1] - A[1][0] * A[0][1]; //<-- cross product z value
}
int determinant3(vector<vector<int>> A)
{
    assert(A.size() == 3);
    assert(A[0].size() == 3);

    //Side note : a b c are the values of the cross product vector if first column of A is 1 1 1 
    int a = A[0][0] * (A[1][1] * A[2][2] - A[2][1] * A[1][2]);
    int b = A[1][0] * (A[2][1] * A[0][2] - A[0][1] * A[2][2]);
    int c = A[2][0] * (A[0][1] * A[1][2] - A[1][1] * A[0][2]);

    return a + b + c;
}
//Addition of two vectors
//Intuition: each vector represent moving towards that vector, up towards positive y, or down towards negative y
//left towareds negative x, and right towards positive x
//The sum of two vectors is the consequence of moving towards the first then the second vector or vice versa.
vector<int> vectorAddition(vector<int> v, vector<int>w)
{
    assert(v.size() == w.size());

    vector<int> ans(v.size());
    for (int i = 0; i < v.size(); i++)
    {
        ans[i] = v[i] + w[i];
    }
    return ans;
}
//Multiplication of vectors (scaling) 
//Intuition: scaling the length of the vector
vector<int> vectorMultiplication(int s, vector<int> v)
{
    for (int i = 0; i < v.size(); i++)
        v[i] *= s;
    return v;
}
//2D Transformation Multiplication of a square matrix and a 2d vector 
//Intuition: imagining the columns of the matrix as transformed basis vectors i^ j^, we want to find where x lands in this transformation
//Note: transformation that keep preserve dot product=0 are called orthonormal (example: rigid transformation like rotation, translation & scaling)
// [a c] [x] = x[a] + y[c] = [xa + yc] 
// [b d] [y]    [b]    [d]   [xb + yd]
vector<int> transform2d(vector<vector<int>> A, vector<int> x)
{
    assert(x.size() == 2);
    assert(A.size() == 2);
    assert(A[0].size() == 2);

    vector<int> result(2);
    result[0] = x[0] * A[0][0] + x[1] * A[1][0];
    result[1] = x[0] * A[1][0] + x[1] * A[1][1];

    return result;
}
/* TODO
//Inverse of a matrix
//Intuition: it is the opposite transformation of the input matrix
vector<vector<int>> inverseOfMatrix(vector<vector<int>> A)
{
    return vector<vector<int>>();
}

//Solving linear of equations Ax = b or matlab A\b
//Intuition: find the original vector after a transformation happened.
vector<int> solveAslashB2(vector<vector<int>> A, vector<int> b)
{
    //Find the inverse of A if the determinant is not zero
    int d = determinant2(A);
    vector<vector<int>> A_;
    if (d != 0)
    {
        A_ = inverseOfMatrix(A);
    }
    else ???
    vector<int> x = transform2d(A_, b);
    return x;
}


//Eigenvector
//Intuition: a nonzero vector that after the transformation A doesn't lose its span but only gets scaled by eigenvalue lambda. For example, in a rotation transformation it's the rotation axis.
//Av = yv --> (A-yI)v = 0
void eigenvectors(vector<vector<int>> A, int &lambda1, int &lambda2, vector<int> &e1, vector<int>& e2)
{
}

//Convolution
//
void convolution2d(vector<vector<int>> A, vector<vector<int>> kernal, bool zeropadding)
{

}

*/

//Create a polygon: triangulate a polygon given number of sides 
//From Udacity interactive computer graphics course (original javascript, c++ not tested!)
void create2DPolygon(int sides, vector<vector<float>> &vertices, vector<vector<int>> &faces)
{
    for (int i = 0; i < sides; i++)
    {
        // Add 90 degrees so we start at +Y axis, rotate counterclockwise around
        double angle = (3.14 / 2.) + (i / sides) * 2. * 3.14;

        float x = cos(angle);
        float y = sin(angle);

        vertices.push_back({ x, y, 0 });
        //Create polygon, all triangles starting at vertex 0
        if (i >= 2)
        {
            faces.push_back({ 0, i - 1, i });
        }
    }
}

/*
  Check if two points are on diagonals or anti diagonals of square board
*/
bool onDiagonal(pair<int, int> c1, pair<int, int> c2)
{
    //check / diagonal (rows+cols) 
    if ((c1.first + c1.second) == (c2.first + c2.second))
        return true;
    //check \ anti-diagonal (rows-cols)
    if ((c1.first - c1.second) == (c2.first - c2.second))
        return true;

    return false;
}