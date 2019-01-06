#include "svm.hpp" 

constexpr auto tol = 1e-6;
auto max = [](auto first, auto second) { return first > second ? first : second};
auto min = [](auto first, auto second) { return first < second ? first : second};

svm::svm()
{}

bool svm::take_step(const linalg::matrix& X, const linalg::vector& y, int i_1, int i_2)
{
	if (i_1 == i_2)
		return false;

	double alpha_1 = this->_alpha[i_1];
	double alpha_2 = this->_alpha[i_2];
	int y_1 = y[i_1];
	int y_2 = y[i_2];
	int s = y_1 * y_2;
	double error_1 = this->_errors[i_1];
	double error_2 = this->_errors[i_2];

	double L = 0.0, H = 0.0;

	if (y_1 != y_2)
	{
		L = max(0, alpha_1 - alpha_2);
		H = min(this->_C, this->_C + alpha_1 - alpha_2);
	}
	else
	{
		L = max(0, alpha_1 + alpha_2 - this->_C);
		H = min(this->_C, alpha_1 + alpha_2);
	}

	if (abs(L - H) < tol)
		return false;

	double K_1_1 = this->_kernel(X[i_1], X[i_1], this->_params);
	double K_1_2 = this->_kernel(X[i_1], X[i_2], this->_params);
	double K_2_2 = this->_kernel(X[i_2], X[i_2], this->_params);
	double eta = 2 * K_1_2 - K_1_1 - K_2_2;

	double a1 = 0.0, a2 = 0.0;

	if (eta < 0)
	{
		a2 = alpha_2 + y_2 * (error_1 - error_2) / eta;

		if (L > a2)
			a2 = L;
		else if (H < a2)
			a2 = H;
	}
	else
	{

	}
}

int svm::examine_example(const linalg::matrix& X, const linalg::vector& y, int i_1)
{
	double y_1 = y[i_1];
	double alpha_1 = this->_alpha[i_1];
	double error_1 = this->_errors[i_1];
	double r_1 = error_1 * y_1;

	if ((r_1 < -tol && alpha_1 < this->_C) || (r_1 > tol && alpha_1 > 0))
	{
		int k, i_2, k0;
		double tmax;

		for (k = 0, i_2 = -1, tmax = 0.0; k < X.cols(); ++k)
		{
			if (this->_alpha[k] > 0 && this->_alpha[k] < this->_C)
			{
				double error_2 = this->_errors[k];
				double temp = abs(error_1 - error_2);

				if (temp > tmax)
				{
					tmax = temp;
					i_2 = k;
				}
			}
			if (i_2 >= 0 && take_step(X,y,i_1,i_2))
				return 1;
		}
	}

	return 0;
}

void svm::fit(const linalg::matrix& X, const linalg::vector& y, double C = HUGE_VAL, double(*kernel)(linalg::vector, linalg::vector, std::initializer_list<double>) = linear, std::initializer_list<double> params = {})
{
	this->_C = C;
	this->_kernel = kernel;
	this->_params = params;
	this->_alpha = linalg::vector(X.cols());
	this->_errors = linalg::vector(X.cols());
	this->_b = 0.0;

	int num_changed = 0;
	bool examine_all = true;

	while (num_changed > 0 && examine_all)
	{
		num_changed = 0;

		if (examine_all)
			for (int i = 0; i < _alpha.length(); ++i)
				num_changed += 1;
		else
			for (int i = 0; i < _alpha.length(); ++i)
				if (this->_alpha[i] != 0 && this->_alpha[i] != this->_C)
					num_changed += 1;

		if (examine_all)
			examine_all = !examine_all;
		else if (num_changed == 0)
			examine_all = !examine_all;
	}

}

double linear(linalg::vector v1, linalg::vector v2, std::initializer_list<double> params = {})
{
	return v1 * v2;
}

double polynomial(linalg::vector v1, linalg::vector v2, std::initializer_list<double> params = {})
{
	return pow(v1 * v2, *(params.begin()));
}

