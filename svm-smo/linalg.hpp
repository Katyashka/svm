#ifndef LINALG_H
#define LINALG_H

#include <exception>

namespace linalg
{
	class vector
	{
	private:
		int _l;
		double* _vec;
	public:
		vector();
		vector(int);
		vector(double*, int);
		vector(const vector&);
		~vector();
		vector& operator=(const vector&);
		double& operator[](int) const;
		int length() const;
	};

	class matrix
	{
	private:
		int _m;
		int _n;
		vector* _mat;
	public:
		matrix();
		matrix(int, int);
		matrix(const vector*, int);
		matrix(const matrix&);
		~matrix();
		matrix& operator=(const matrix&);
		const vector& operator[](int) const;
		int cols() const;
		int rows() const;
	};

	double operator*(const vector&, const vector&);
	matrix operator~(const matrix&);
	matrix operator*(const matrix&, const matrix&);
	matrix operator*(const matrix&, const double&);
	matrix operator*(const double&, const matrix&);
	matrix operator+(const matrix&, const matrix&);
	vector operator*(const vector&, const matrix&);
	vector operator*(const double&, const vector&);
	vector operator*(const vector&, const double&);
	vector operator*(const matrix&, const vector&);
	vector operator+(const vector&, const vector&);
}

#endif