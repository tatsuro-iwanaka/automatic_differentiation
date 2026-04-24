# autodiff

A header-only C++ library for Forward-Mode Automatic Differentiation (AD). 
This library provides a dual number (`dual<T>`) implementation and a fully AD-compatible complex number (`complex<T>`) structure.

## Features

* **Header-only**
* **Forward-Mode AD**: Computes exact derivatives (Jacobians) without the truncation or round-off errors associated with finite difference methods.
* **First-Class Complex Number Support**: Native implementation of `complex<T>` that flawlessly interacts with `dual<T>`, enabling the differentiation of algorithms in the complex plane.
* **Comprehensive `<cmath>` Coverage**: Support for trigonometric, hyperbolic, exponential, logarithmic, and error functions (`erf`, `erfc`) using robust Two-Step ADL (Argument-Dependent Lookup).

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
	autodiff::dual<double> y = x * x * sin(x);
		
	std::cout << "f(x)  = " << y.val << std::endl;
	std::cout << "f'(x) = " << y.der << std::endl; // Analytically exact derivative
	std::cout << "exact f'(x) = " << 2.0 * 5.0 * std::sin(5.0) + 5.0 * 5.0 * std::cos(5.0) << std::endl;
		
	return 0;
}
