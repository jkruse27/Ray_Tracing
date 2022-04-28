// Codigo adaptado de https://raytracing.github.io/books/RayTracingInOneWeekend.html


//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#pragma once

#include <cmath>
#include <iostream>
#include <string>
#include <sstream>

using std::sqrt;
using std::fabs;

class color {
    public:
        color() : e{0,0,0} {}
        color(double e0, double e1, double e2) : e{e0, e1, e2} {}

        double r() const { return e[0]; }
        double g() const { return e[1]; }
        double b() const { return e[2]; }

        color operator-() const { return color(-e[0], -e[1], -e[2]); }
        double operator[](int i) const { return e[i]; }
        double& operator[](int i) { return e[i]; }

        color& operator+=(const color &v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }

        color& operator*=(const double t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }

       color& operator/=(const double t) {
            return *this *= 1/t;
        }

        double length() const {
            return sqrt(length_squared());
        }

        double length_squared() const {
            return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
        }
        std::string color_text() {
            std::stringstream ss;

            // Write the translated [0,255] value of each color component.
            ss << static_cast<int>(255.999 * this->r()) << ' '
                << static_cast<int>(255.999 * this->g()) << ' '
                << static_cast<int>(255.999 * this->b());

            std::string s = ss.str();
            return s;
}
    public:
        double e[3];
};

// color Utility Functions

inline std::ostream& operator<<(std::ostream &out, const color &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline color operator+(const color &u, const color &v) {
    return color(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline color operator-(const color &u, const color &v) {
    return color(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline color operator*(const color &u, const color &v) {
    return color(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline color operator*(double t, const color &v) {
    return color(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline color operator*(const color &v, double t) {
    return t * v;
}

inline color operator/(color v, double t) {
    return (1/t) * v;
}

inline double dot(const color &u, const color &v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

inline color cross(const color &u, const color &v) {
    return color(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline color unit_vector(color v) {
    return v / v.length();
}