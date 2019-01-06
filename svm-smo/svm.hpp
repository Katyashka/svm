#ifndef SVM_H
#define SVM_H

#include "linalg.hpp"
#include <vector>
#include <initializer_list>

class svm
{
private:
	double _b;
	linalg::vector _alpha;
	double _C;
	double (*_kernel)(linalg::vector, linalg::vector, std::initializer_list<double>);
	std::initializer_list<double> _params;
	linalg::vector _errors;
	int examine_example(const linalg::matrix&, const linalg::vector&, int);
	bool take_step(const linalg::matrix&, const linalg::vector&, int, int);

public:
	svm();
	void fit(const linalg::matrix&, const linalg::vector&, double, double(*kernel)(linalg::vector, linalg::vector, std::initializer_list<double>), std::initializer_list<double>);
	linalg::vector predict(const linalg::matrix&);
};

double linear(linalg::vector, linalg::vector, std::initializer_list<double>);
double polynomial(linalg::vector, linalg::vector, std::initializer_list<double>);

#endif