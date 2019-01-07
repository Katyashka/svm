#ifndef SVM_H
#define SVM_H

#include "linalg.hpp"
#include <initializer_list>

namespace kernels
{
	double linear(const linalg::vector&, const linalg::vector&, std::initializer_list<double>);
	double polynomial(const linalg::vector&, const linalg::vector&, std::initializer_list<double>);
	double gaussian(const linalg::vector&, const linalg::vector&, std::initializer_list<double>);
	double sigmoid(const linalg::vector&, const linalg::vector&, std::initializer_list<double>);
}

class svm
{
public:
	std::initializer_list<double> _params;
	linalg::matrix _X;
	linalg::vector _y;
	linalg::vector _w;
	linalg::vector _alpha;
	linalg::vector _errors;
	double _b;
	double _C;
	double(*_kernel)(const linalg::vector&, const linalg::vector&, std::initializer_list<double>);
	int examine_example(int);
	bool take_step(int, int);
	double objective_function(const linalg::vector&);
	svm();
	void train(const linalg::matrix&, const linalg::vector&, double = 1e+300, double(*kernel)(const linalg::vector&, const linalg::vector&, std::initializer_list<double>) = kernels::linear, std::initializer_list<double> = { 0.0 });
	linalg::vector predict(const linalg::matrix&);
};

#endif