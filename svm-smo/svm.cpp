#include "svm.hpp" 

constexpr auto tol = 1e-6;
constexpr auto eps = tol;
const auto max = [](auto first, auto second) { return first > second ? first : second; };
const auto min = [](auto first, auto second) { return first < second ? first : second; };

svm::svm()
{}

double svm::objective_function(const linalg::vector& x)
{
	double res = 0.0;

	for (int i = 0; i < this->_X.rows(); ++i)
		res += this->_alpha[i] * this->_y[i] * this->_kernel(this->_X[i], x, this->_params);

	return res - this->_b;
}

bool svm::take_step(int i_1, int i_2)
{
	if (i_1 == i_2)
		return false;

	double alpha_1 = this->_alpha[i_1];
	double alpha_2 = this->_alpha[i_2];
	int y_1 = this->_y[i_1];
	int y_2 = this->_y[i_2];
	int s = y_1 * y_2;
	double error_1 = this->objective_function(this->_X[i_1]) - this->_y[i_1];
	double error_2 = this->objective_function(this->_X[i_2]) - this->_y[i_2];

	double L = 0.0, H = 0.0;

	if (y_1 != y_2)
	{
		L = max(0, alpha_2 - alpha_1);
		H = min(this->_C, this->_C + alpha_2 - alpha_1);
	}
	else
	{
		L = max(0, alpha_1 + alpha_2 - this->_C);
		H = min(this->_C, alpha_1 + alpha_2);
	}

	if (abs(L - H) < tol)
		return false;

	double K_1_1 = this->_kernel(this->_X[i_1], this->_X[i_1], this->_params);
	double K_1_2 = this->_kernel(this->_X[i_1], this->_X[i_2], this->_params);
	double K_2_2 = this->_kernel(this->_X[i_2], this->_X[i_2], this->_params);
	double eta = 2 * K_1_2 - K_1_1 - K_2_2;

	double a_2 = 0.0;

	if (eta < 0)
	{
		a_2 = alpha_2 - y_2 * (error_1 - error_2) / eta;

		if (L > a_2)
			a_2 = L;
		else if (H < a_2)
			a_2 = H;
	}
	else
	{
		double c_1 = eta / 2.0;
		double c_2 = y_2 * (error_1 - error_2) - eta * alpha_2;
		double L_obj = c_1 * L * L + c_2 * L;
		double H_obj = c_1 * H * H + c_2 * H;

		if (L_obj > H_obj + eps)
			a_2 = L;
		else if (L_obj < H_obj - eps)
			a_2 = H;
		else
			a_2 = alpha_2;
	}

	if (a_2 < eps)
		a_2 = 0;
	else if (a_2 > this->_C - eps)
		a_2 = this->_C;

	if (abs(a_2 - alpha_2) < eps * (a_2 + alpha_2 + eps))
		return false;

	double a_1 = alpha_1 + s * (alpha_2 - a_2);

	if (a_1 < eps)
		a_1 = 0;
	else if (a_1 > this->_C - eps)
		a_1 = this->_C;

	double b_1 = error_1 + y_1 * (a_1 - alpha_1) * K_1_1 + y_2 * (a_2 - alpha_2) * K_1_2 + this->_b;
	double b_2 = error_2 + y_1 * (a_1 - alpha_1) * K_1_2 + y_2 * (a_2 - alpha_2) * K_2_2 + this->_b;
	double b_new = 0.0;

	if (a_1 > 0 && a_1 < this->_C)
		b_new = b_1;
	else if (a_2 > 0 && a_2 < this->_C)
		b_new = b_2;
	else
		b_new = 0.5 * (b_1 + b_2);

	this->_alpha[i_1] = a_1;
	this->_alpha[i_2] = a_2;

	for (int i = 0; i < this->_errors.length(); ++i)
		if (i != i_1 && i != i_2)
			this->_errors[i] += y_1 * (a_1 - alpha_1) * this->_kernel(this->_X[i_1], this->_X[i], this->_params) + y_2 * (a_2 - alpha_2) * this->_kernel(this->_X[i_2], this->_X[i], this->_params);

	if (a_2 > 0 && a_2 < this->_C)
		this->_errors[i_2] = 0.0;
	if (a_1 > 0 && a_1 < this->_C)
		this->_errors[i_1] = 0.0;

	return true;
}

int svm::examine_example(int i_1)
{
	double y_1 = this->_y[i_1];
	double alpha_1 = this->_alpha[i_1];
	double error_1 = this->objective_function(this->_X[i_1]) - this->_y[i_1];
	double r_1 = error_1 * y_1;

	if ((r_1 < -tol && alpha_1 < this->_C) || (r_1 > tol && alpha_1 > 0))
	{
		int k, i_2, k_0;
		double tmax;

		for (k = 0, i_2 = -1, tmax = 0.0; k < this->_alpha.length(); ++k)
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
			if (i_2 >= 0 && this->take_step(i_1, i_2))
				return 1;
		}

		for (k_0 = rand() % this->_alpha.length(), k = k_0; k < this->_alpha.length() + k_0; ++k)
		{
			i_2 = k % this->_alpha.length();

			if (this->_alpha[i_2] > 0 && this->_alpha[i_2] < this->_C && this->take_step(i_1, i_2))
				return 1;
		}

		for (k_0 = rand() % this->_alpha.length(), k = k_0; k < this->_alpha.length() + k_0; ++k)
		{
			i_2 = k % this->_alpha.length();

			if (this->take_step(i_1, i_2))
				return 1;
		}
	}

	return 0;
}

void svm::train(const linalg::matrix& X, const linalg::vector& y, double C, double(*kernel)(const linalg::vector&, const linalg::vector&, std::initializer_list<double>), std::initializer_list<double> params)
{
	this->_X = X;
	this->_y = y;
	this->_C = C;
	this->_kernel = kernel;
	this->_params = params;
	this->_alpha = linalg::vector(X.rows());
	this->_b = 0.0;
	this->_errors = linalg::vector(X.rows());

	for (int i = 0; i < this->_errors.length(); ++i)
		this->_errors[i] = this->objective_function(X[i]) - y[i];

	int num_changed = 0;
	bool examine_all = true;

	while (num_changed > 0 || examine_all)
	{
		num_changed = 0;

		if (examine_all)
			for (int i = 0; i < this->_alpha.length(); ++i)
				num_changed += this->examine_example(i);
		else
			for (int i = 0; i < this->_alpha.length(); ++i)
				if (this->_alpha[i] != 0 && this->_alpha[i] != this->_C)
					num_changed += this->examine_example(i);;

		if (examine_all)
			examine_all = !examine_all;
		else if (num_changed == 0)
			examine_all = !examine_all;
	}

	this->_w = linalg::vector(X.cols());

	for (int i = 0; i < this->_w.length(); ++i)
		this->_w = this->_w + this->_alpha[i] * this->_y[i] * this->_X[i];
}

linalg::vector svm::predict(const linalg::matrix& X)
{
	linalg::vector y(X.rows());

	for (int i = 0; i < y.length(); ++i)
		y[i] = this->_w * X[i] - this->_b;

	return y;
}

double kernels::linear(const linalg::vector& v1, const linalg::vector& v2, std::initializer_list<double> params = { 0.0 })
{
	return v1 * v2 + *(params.begin());
}

double kernels::polynomial(const linalg::vector& v1, const linalg::vector& v2, std::initializer_list<double> params = { 0.0, 2.0 })
{
	return pow(kernels::linear(v1, v2, { *(params.begin()) }), *(params.begin() + 1));
}

double kernels::gaussian(const linalg::vector& v1, const linalg::vector& v2, std::initializer_list<double> params = { 1.0 })
{
	return exp(-((v1 - v2) * (v1 - v2)) / (2 * *(params.begin()) * *(params.begin())));
}

double kernels::sigmoid(const linalg::vector& v1, const linalg::vector& v2, std::initializer_list<double> params = { 0.0, 0.0 })
{
	return tanh(*(params.begin()) + *(params.begin() + 1) * (v1 * v2));
}