# autodiff

A header-only C++ library for Forward-Mode Automatic Differentiation (AD). 
This library provides a dual number (`dual<T>`) implementation and a fully AD-compatible complex number (`complex<T>`) structure.

## Usage

### Basic Automatic Differentiation

To compute the derivative of a function $f(x) = x^2 \sin(x)$ at $x = 5.0$ for example:

```cpp
#include <iostream>
#include <cmath>
#include "autodiff.hpp"

int main(void)
{
	// Initialize x with value 5.0 and derivative seed 1.0 (dx/dx = 1)
	autodiff::dual<double> x(5.0, 1.0);
		
	// Perform operations
	autodiff::dual<double> y = x * x * autodiff::sin(x);
		
	std::cout << "f(x)  = " << y.val << std::endl;
	std::cout << "f'(x) = " << y.der << std::endl; // Analytically exact derivative
	std::cout << "exact f'(x) = " << 2.0 * 5.0 * std::sin(5.0) + 5.0 * 5.0 * std::cos(5.0) << std::endl;
		
	return 0;
}
