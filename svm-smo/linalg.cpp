#include "linalg.hpp"

namespace linalg
{
	vector::vector()
	{
		this->_l = 0;
		this->_vec = new double[0];
	}

	vector::vector(int l)
	{
		this->_l = l;
		this->_vec = new double[l];

		for (int i = 0; i < l; ++i)
			this->_vec[i] = 0;
	}

	vector::vector(double* v, int l)
	{
		this->_l = l;
		this->_vec = new double[l];

		for (int i = 0; i < l; ++i)
			this->_vec[i] = v[i];
	}

	vector::vector(const vector& v)
	{
		this->_l = v._l;
		this->_vec = new double[v._l];

		for (int i = 0; i < v._l; ++i)
			this->_vec[i] = v[i];
	}

	vector::~vector()
	{
		delete[] _vec;
		this->_vec = nullptr;
	}

	vector& vector::operator=(const vector& v)
	{
		delete[] this->_vec;
		this->_l = v._l;
		this->_vec = new double[v._l];

		for (int i = 0; i < v._l; ++i)
			this->_vec[i] = v[i];

		return *this;
	}

	double& vector::operator[](int i) const
	{
		if (i >= this->_l)
			throw std::exception("Wrong size");

		return this->_vec[i];
	}

	int vector::length() const
	{
		return this->_l;
	}

	double operator*(const vector& v1, const vector& v2)
	{
		if (v1.length() != v2.length())
			throw std::exception("Wrong sizes");

		double res = 0.0;

		for (int i = 0; i < v1.length(); ++i)
			res += v1[i] * v2[i];

		return res;
	}

	vector operator+(const vector& v1, const vector& v2)
	{
		if (v1.length() != v2.length())
			throw std::exception("Wrong sizes");

		vector res(v1.length());

		for (int i = 0; i < v2.length(); ++i)
			res[i] = v1[i] + v2[i];

		return res;
	}

	vector operator*(const double& k, const vector& v)
	{
		vector res(v.length());

		for (int i = 0; i < v.length(); ++i)
			res[i] = v[i] * k;

		return res;
	}

	vector operator*(const vector& v, const double& k)
	{
		vector res(v.length());

		for (int i = 0; i < v.length(); ++i)
			res[i] = v[i] * k;

		return res;
	}

	matrix::matrix()
	{
		this->_m = 0;
		this->_n = 0;
		this->_mat = new vector[0];
	}

	matrix::matrix(int m, int n)
	{
		this->_m = m;
		this->_n = n;
		this->_mat = new vector[m];

		for (int i = 0; i < m; ++i)
			this->_mat[i] = vector(n);
	}

	matrix::matrix(const vector* m, int l)
	{
		this->_m = l;
		this->_n = m->length();
		this->_mat = new vector[l];

		for (int i = 0; i < l; ++i)
			this->_mat[i] = m[i];
	}

	matrix::matrix(const matrix& m)
	{
		this->_m = m._m;
		this->_n = m._n;
		this->_mat = new vector[this->_m];

		for (int i = 0; i < this->_m; ++i)
			this->_mat[i] = vector(this->_n);

		for (int i = 0; i < this->_m; ++i)
			for (int j = 0; j < this->_n; ++j)
				this->_mat[i][j] = m[i][j];
	}

	matrix::~matrix()
	{
		this->_mat = nullptr;
	}

	matrix& matrix::operator=(const matrix& m)
	{
		if (&m == this)
			return *this;

		for (int i = 0; i < this->rows(); ++i)
			this->_mat->~vector();

		this->_mat = nullptr;

		this->_m = m._m;
		this->_n = m._n;
		this->_mat = new vector[this->_m];

		for (int i = 0; i < this->_m; ++i)
			this->_mat[i] = vector(this->_n);

		for (int i = 0; i < this->_m; ++i)
			for (int j = 0; j < this->_n; ++j)
				this->_mat[i][j] = m._mat[i][j];

		return *this;
	}

	const vector& matrix::operator[](int i) const
	{
		if (i >= this->_m)
			throw std::exception("Wrong size");

		return this->_mat[i];
	}

	int matrix::rows() const
	{
		return this->_m;
	}

	int matrix::cols() const
	{
		return this->_n;
	}

	matrix operator~(const matrix& m)
	{
		matrix res(m.cols(), m.rows());

		for (int i = 0; i < res.rows(); ++i)
			for (int j = 0; j < res.cols(); ++j)
				res[i][j] = m[j][i];

		return res;
	}

	matrix operator*(const matrix& m1, const matrix& m2)
	{
		if (m1.rows() != m2.rows())
			throw std::exception("Wrong sizes");

		matrix res(m1.rows(), m2.cols());

		for (int i = 0; i < res.rows(); ++i)
			for (int j = 0; j < res.cols(); ++j)
			{
				double e = 0.0f;

				for (int k = 0; k < m2.rows(); ++k)
					e += m1[i][k] * m2[k][j];

				res[i][j] = e;
			}

		return res;
	}

	matrix operator*(const matrix& m, const double& k)
	{
		matrix res(m.rows(), m.cols());

		for (int i = 0; i < res.rows(); ++i)
			for (int j = 0; j < res.cols(); ++j)
				res[i][j] = m[i][j] * k;

		return res;
	}

	matrix operator*(const double& k, const matrix& m)
	{
		matrix res(m.rows(), m.cols());

		for (int i = 0; i < res.rows(); ++i)
			for (int j = 0; j < res.cols(); ++j)
				res[i][j] = m[i][j] * k;

		return res;
	}


	matrix operator+(const matrix& m1, const matrix& m2)
	{
		if (m1.rows() != m2.rows() && m1.cols() != m2.cols())
			throw std::exception("Wrong sizes");

		matrix res(m1.rows(), m2.cols());

		for (int i = 0; i < res.rows(); ++i)
			for (int j = 0; j < res.cols(); ++j)
				res[i][j] = m1[i][j] + m2[i][j];

		return res;
	}

	vector operator*(const vector& v, const matrix& m)
	{
		if (v.length() != m.rows())
			throw std::exception("Wrong sizes");

		vector res(m.cols());

		for (int i = 0; i < m.rows(); ++i)
			for (int j = 0; j < m.cols(); ++j)
				res[j] += v[i] * m[i][j];

		return res;
	}

	vector operator*(const matrix& m, const vector& v)
	{
		if (v.length() != m.cols())
			throw std::exception("Wrong sizes");

		vector res(m.rows());

		for (int i = 0; i < m.cols(); ++i)
			for (int j = 0; j < m.rows(); ++j)
				res[j] += v[i] * m[i][j];

		return res;
	}
}