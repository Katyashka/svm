#ifndef SVM_H
#define SVM_H

#include "linalg.hpp"
#include <vector>

class svm
{
private:
	linalg::vector _w;
	double _b;
	std::vector<int> _supports;
	double _C;
	double (*_kernel)(linalg::vector, linalg::vector, double...);
	std::vector<double> _params;

public:
	svm(double, double(*)(linalg::vector, linalg::vector, double...), double...);
	void fit(const linalg::matrix&, const linalg::vector&);
	linalg::vector predict(const linalg::matrix&);
	linalg::vector _w();
	double _b();
	std::vector<int> _supports();
};

#endif