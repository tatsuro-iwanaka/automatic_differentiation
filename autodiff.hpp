#pragma once

#include <iostream>
#include <cmath>

namespace autodiff
{
// 双対数構造体
template <typename T> struct dual
{
	T val; // 実数部
	T der; // 双対部

	// コンストラクタ
	dual(T v = 0.0, T d = 0.0) : val(v), der(d)
	{
		;
	}

	dual operator+(const dual&) const;
	dual operator-(const dual&) const;
	dual operator-() const;
	dual operator*(const dual&) const;
	dual operator/(const dual&) const;

	dual& operator+=(const dual<T>&);
	dual& operator-=(const dual<T>&);
	dual& operator*=(const dual<T>&);
	dual& operator/=(const dual<T>&);

	dual& operator+=(const T&);
	dual& operator-=(const T&);
	dual& operator*=(const T&);
	dual& operator/=(const T&);
};

template <typename T> inline dual<T> dual<T>::operator+(const dual<T>& rhs) const
{
	return dual<T>(val + rhs.val, der + rhs.der);
}

template <typename T> inline dual<T> dual<T>::operator-(const dual<T>& rhs) const
{
	return dual<T>(val - rhs.val, der - rhs.der);
}

template <typename T> inline dual<T> dual<T>::operator-() const
{
	return dual<T>(-val, -der);
}

template <typename T> inline dual<T> dual<T>::operator*(const dual<T>& rhs) const
{
	return dual<T>(val * rhs.val, val * rhs.der + der * rhs.val);
}

template <typename T> inline dual<T> dual<T>::operator/(const dual<T>& rhs) const
{
	return dual<T>(val / rhs.val, (der * rhs.val - val * rhs.der) / (rhs.val * rhs.val));
}

template <typename T> inline dual<T> sin(const dual<T>& x)
{
	return dual<T>(std::sin(x.val), std::cos(x.val) * x.der);
}

template <typename T> inline dual<T> cos(const dual<T>& x)
{
	return dual<T>(std::cos(x.val), -std::sin(x.val) * x.der);
}

template <typename T> inline dual<T> tan(const dual<T>& x)
{
	return dual<T>(std::tan(x.val), x.der / std::cos(x.val) / std::cos(x.val));
}

template <typename T> inline dual<T> asin(const dual<T>& x)
{
	// d/dx(asin(x)) = 1 / sqrt(1 - x^2)
	return dual<T>(std::asin(x.val), x.der / std::sqrt(T(1.0) - x.val * x.val));
}

template <typename T> inline dual<T> acos(const dual<T>& x)
{
	// d/dx(acos(x)) = -1 / sqrt(1 - x^2)
	return dual<T>(std::acos(x.val), -x.der / std::sqrt(T(1.0) - x.val * x.val));
}

template <typename T> inline dual<T> atan(const dual<T>& x)
{
	// d/dx(atan(x)) = 1 / (1 + x^2)
	return dual<T>(std::atan(x.val), x.der / (T(1.0) + x.val * x.val));
}

template <typename T> inline dual<T> atan2(const dual<T>& y, const dual<T>& x)
{
	// 実部の計算
	T val = std::atan2(y.val, x.val);
	
	// 分母 (x^2 + y^2) の計算
	T denom = x.val * x.val + y.val * y.val;
	
	// 微分成分（双対部）の計算
	// 連鎖律: (dy * x - dx * y) / (x^2 + y^2)
	T der = (y.der * x.val - x.der * y.val) / denom;
	
	return dual<T>(val, der);
}

template <typename T> inline dual<T> sinh(const dual<T>& x)
{
	return dual<T>(std::sinh(x.val), std::cosh(x.val) * x.der);
}

template <typename T> inline dual<T> cosh(const dual<T>& x)
{
	return dual<T>(std::cosh(x.val), std::sinh(x.val) * x.der);
}

template <typename T> inline dual<T> tanh(const dual<T>& x)
{
	T th = std::tanh(x.val);
	return dual<T>(th, x.der * (T(1.0) - th * th));
}

template <typename T> inline dual<T> exp(const dual<T>& x)
{
	T e = std::exp(x.val);
	return dual<T>(e, e * x.der);
}

template <typename T> inline dual<T> log(const dual<T>& x)
{
	return dual<T>(std::log(x.val), x.der / x.val);
}

template <typename T> inline dual<T> floor(const dual<T>& x)
{
	return dual<T>(std::floor(x.val), 0.0);
}

template <typename T> inline dual<T> cbrt(const dual<T>& x)
{
	return dual<T>(std::cbrt(x.val), x.der / (3.0 * std::pow(x.val, 2.0 / 3.0)));
}

template <typename T> inline dual<T> abs(const dual<T>& x)
{
	T sign = (x.val > 0.0) ? 1.0 : ((x.val < 0.0) ? -1.0 : 0.0);
	return dual<T>(std::abs(x.val), x.der * sign);
}

template <typename T> inline dual<T> sqrt(const dual<T>& x)
{
	return dual<T>(std::sqrt(x.val), x.der / (2.0 * std::sqrt(x.val)));
}

template <typename T> inline dual<T> pow(const dual<T>& base, const dual<T>& exp)
{
	// dual^dual : d/dx(u^v) = v*u^(v-1)*u' + u^v*ln(u)*v'
	T val = std::pow(base.val, exp.val);
	T der = exp.val * std::pow(base.val, exp.val - T(1.0)) * base.der + val * std::log(base.val) * exp.der;
	return dual<T>(val, der);
}

template <typename T> inline dual<T> pow(const dual<T>& base, const T& exp)
{
	// dual^scalar
	return pow(base, dual<T>(exp, T(0.0)));
}

template <typename T> inline dual<T> pow(const T& base, const dual<T>& exp)
{
	// scalar^dual
	return pow(dual<T>(base, T(0.0)), exp);
}

template <typename T> inline dual<T> erf(const dual<T>& x)
{
	T val = std::erf(x.val);
	T der = x.der * T(2.0) / std::sqrt(T(std::numbers::pi)) * std::exp(-x.val * x.val);
	return dual<T>(val, der);
}

template <typename T> inline dual<T> erfc(const dual<T>& x)
{
	T val = std::erfc(x.val);
	T der = -x.der * T(2.0) / std::sqrt(T(std::numbers::pi)) * std::exp(-x.val * x.val);
	return dual<T>(val, der);
}

template <typename T> inline dual<T> operator+(const T& lhs, const dual<T>& rhs)
{
	return dual<T>(lhs) + rhs;
}

template <typename T> inline dual<T> operator-(const T& lhs, const dual<T>& rhs)
{
	return dual<T>(lhs) - rhs;
}

template <typename T> inline dual<T> operator*(const T& lhs, const dual<T>& rhs)
{
	return dual<T>(lhs) * rhs;
}

template <typename T> inline dual<T> operator/(const T& lhs, const dual<T>& rhs)
{
	return dual<T>(lhs) / rhs;
}

template <typename T> inline bool operator<(const dual<T>& lhs, const dual<T>& rhs)
{
	return lhs.val < rhs.val;
}

template <typename T> inline bool operator>(const dual<T>& lhs, const dual<T>& rhs)
{
	return lhs.val > rhs.val;
}

template <typename T> inline bool operator<=(const dual<T>& lhs, const dual<T>& rhs)
{
	return lhs.val <= rhs.val;
}

template <typename T> inline bool operator>=(const dual<T>& lhs, const dual<T>& rhs)
{
	return lhs.val >= rhs.val;
}

template <typename T> inline bool operator==(const dual<T>& lhs, const dual<T>& rhs)
{
	return lhs.val == rhs.val;
}

template <typename T> inline bool operator!=(const dual<T>& lhs, const dual<T>& rhs)
{
	return lhs.val != rhs.val;
}

template <typename T> inline std::ostream& operator<<(std::ostream& os, const dual<T>& d)
{
	os << "(" << d.val << ", " << d.der << ")";
	
	return os;
}

template <typename T> inline dual<T>& dual<T>::operator+=(const dual<T>& rhs)
{
	val += rhs.val; der += rhs.der;
	return *this;
}

template <typename T> inline dual<T>& dual<T>::operator-=(const dual<T>& rhs)
{
	val -= rhs.val; der -= rhs.der;
	return *this;
}

template <typename T> inline dual<T>& dual<T>::operator*=(const dual<T>& rhs)
{
	der = val * rhs.der + der * rhs.val; val *= rhs.val;
	return *this;
}

template <typename T> inline dual<T>& dual<T>::operator/=(const dual<T>& rhs)
{
	der = (der * rhs.val - val * rhs.der) / (rhs.val * rhs.val); val /= rhs.val;
	return *this;
}

template <typename T> inline dual<T>& dual<T>::operator+=(const T& rhs)
{
	val += rhs; return *this;
}

template <typename T> inline dual<T>& dual<T>::operator-=(const T& rhs)
{
	val -= rhs; return *this;
}

template <typename T> inline dual<T>& dual<T>::operator*=(const T& rhs)
{
	val *= rhs; der *= rhs; return *this;
}

template <typename T> inline dual<T>& dual<T>::operator/=(const T& rhs)
{
	val /= rhs; der /= rhs; return *this;
}

template <typename T> inline dual<T> fmax(const dual<T>& x, const dual<T>& y)
{
	return (x.val > y.val) ? x : y;
}

template <typename T> inline dual<T> fmax(const dual<T>& x, const T& y)
{
	return (x.val > y) ? x : dual<T>(y, 0.0);
}

template <typename T> inline dual<T> fmin(const dual<T>& x, const dual<T>& y)
{
	return (x.val < y.val) ? x : y;
}

template <typename T> inline dual<T> fmin(const dual<T>& x, const T& y)
{
	return (x.val < y) ? x : dual<T>(y, 0.0);
}

template <typename T> inline dual<T> ceil(const dual<T>& x)
{
	return dual<T>(std::ceil(x.val), T(0.0));
}

template <typename T> inline dual<T> round(const dual<T>& x)
{
	return dual<T>(std::round(x.val), T(0.0));
}

template <typename T> inline bool isnan(const dual<T>& x)
{
	using std::isnan;
	return isnan(x.val);
}

template <typename T> inline bool isinf(const dual<T>& x)
{
	using std::isinf;
	return isinf(x.val);
}

// 複素数構造体
template <typename T> struct complex
{
	T re;
	T im;

	complex(T r = T(0.0), T i = T(0.0)) : re(r), im(i)
	{
		;
	}

	T real() const;
	T imag() const;
	complex operator+(const complex&) const;
	complex operator-(const complex&) const;
	complex operator-() const;
	complex operator*(const complex&) const;
	complex operator/(const complex&) const;

	complex& operator+=(const complex&);
	complex& operator-=(const complex&);
	complex& operator*=(const complex&);
	complex& operator/=(const complex&);

	complex& operator+=(const T&);
	complex& operator-=(const T&);
	complex& operator*=(const T&);
	complex& operator/=(const T&);
};

template <typename T> inline T complex<T>::real() const
{
	return re;
}

template <typename T> inline T complex<T>::imag() const
{
	return im;
}

template <typename T> inline complex<T> complex<T>::operator+(const complex& o) const
{
	return complex(re + o.re, im + o.im);
}

template <typename T> inline complex<T> complex<T>::operator-(const complex& o) const
{
	return complex(re - o.re, im - o.im);
}

template <typename T> inline complex<T> complex<T>::operator*(const complex& o) const
{ 
	return complex(re * o.re - im * o.im, re * o.im + im * o.re); 
}

template <typename T> inline complex<T> complex<T>::operator/(const complex& o) const
{
	T denom = o.re * o.re + o.im * o.im;
	return complex((re * o.re + im * o.im) / denom, (im * o.re - re * o.im) / denom);
}

template <typename T> inline complex<T> complex<T>::operator-() const
{
	return complex(-re, -im);
}

template <typename T> std::ostream& operator<<(std::ostream& os, const complex<T>& c)
{
	os << "(" << c.re << ", " << c.im << ")";
	return os;
}

template <typename T> complex<T> operator+(const T& a, const complex<T>& b)
{
	return complex<T>(a + b.re, b.im);
}

template <typename T> complex<T> operator-(const T& a, const complex<T>& b)
{
	return complex<T>(a - b.re, -b.im);
}

template <typename T> complex<T> operator*(const T& a, const complex<T>& b)
{
	return complex<T>(a * b.re, a * b.im);
}

template <typename T> complex<T> operator/(const T& a, const complex<T>& b)
{
	T denom = b.re * b.re + b.im * b.im;
	return complex<T>(a * b.re / denom, -a * b.im / denom);
}

template <typename T> complex<T> operator+(const complex<T>& a, const T& b)
{
	return complex<T>(a.re + b, a.im);
}

template <typename T> complex<T> operator-(const complex<T>& a, const T& b)
{
	return complex<T>(a.re - b, a.im);
}

template <typename T> complex<T> operator*(const complex<T>& a, const T& b)
{
	return complex<T>(a.re * b, a.im * b);
}

template <typename T> complex<T> operator/(const complex<T>& a, const T& b)
{
	return complex<T>(a.re / b, a.im / b);
}

template <typename T> inline T abs(const complex<T>& c)
{
	using std::sqrt;
	return sqrt(c.re * c.re + c.im * c.im);
}

template <typename T> inline T norm(const complex<T>& c)
{
	return c.re * c.re + c.im * c.im;
}

template <typename T> inline complex<T> pow(const complex<T>& base, const complex<T>& exponent)
{
	// complex^complex : z^w = exp(w * log(z))
	return exp(exponent * log(base));
}

template <typename T> inline complex<T> pow(const complex<T>& base, const T& exponent)
{
	// complex^scalar
	return pow(base, complex<T>(exponent, T(0.0)));
}

template <typename T> inline complex<T> pow(const T& base, const complex<T>& exponent)
{
	// scalar^complex
	return pow(complex<T>(base, T(0.0)), exponent);
}

template <typename T> inline complex<T> conj(const complex<T>& c)
{
	return complex<T>(c.re, -c.im);
}

template <typename T> inline T arg(const complex<T>& c)
{
	using std::atan2; // std::atan2 を候補に入れる
	auto result = atan2(c.im, c.re);
	return result;
}

template <typename T> complex<T> sin(const complex<T>& z)
{
	using std::sin; using std::cos; using std::sinh; using std::cosh;

	return complex<T>(sin(z.re) * cosh(z.im), cos(z.re) * sinh(z.im));
}

template <typename T> complex<T> cos(const complex<T>& z)
{
	using std::sin; using std::cos; using std::sinh; using std::cosh;

	return complex<T>(cos(z.re) * cosh(z.im), -sin(z.re) * sinh(z.im));
}

template <typename T> complex<T> tan(const complex<T>& z)
{
	using std::sin; using std::cos; using std::sinh; using std::cosh;

	T sx = sin(z.re);
	T cx = cos(z.re);
	T shy = sinh(z.im);
	T chy = cosh(z.im);
	
	complex<T> sinz(sx * chy, cx * shy);
	complex<T> cosz(cx * chy, -sx * shy);
	
	return sinz / cosz;
}

template <typename T> inline complex<T> sinh(const complex<T>& z)
{
	using std::sinh; using std::cosh; using std::sin; using std::cos;

	return complex<T>(sinh(z.re) * cos(z.im), cosh(z.re) * sin(z.im));
}

template <typename T> inline complex<T> cosh(const complex<T>& z)
{
	using std::sinh; using std::cosh; using std::sin; using std::cos;

	return complex<T>(cosh(z.re) * cos(z.im), sinh(z.re) * sin(z.im));
}

template <typename T> inline complex<T> tanh(const complex<T>& z)
{
	return sinh(z) / cosh(z);
}

template <typename T> complex<T> exp(const complex<T>& z)
{
	using std::exp; using std::sin; using std::cos;

	T ex = exp(z.re);
	return complex<T>(ex * cos(z.im), ex * sin(z.im));
}

template <typename T> inline complex<T> log(const complex<T>& z)
{
	using std::log;
	// norm(z) = re^2 + im^2
	// arg(z) = atan2(im, re) 
	T half = T(0.5);
	return complex<T>(half * log(norm(z)), arg(z));
}

template <typename T> bool operator==(const complex<T>& a, const complex<T>& b)
{
	return (a.re == b.re) && (a.im == b.im);
}

template <typename T> bool operator!=(const complex<T>& a, const complex<T>& b)
{
	return !(a == b);
}

template <typename T> inline complex<T>& complex<T>::operator+=(const complex<T>& rhs)
{
	re += rhs.re; im += rhs.im;
	return *this;
}

template <typename T> inline complex<T>& complex<T>::operator-=(const complex<T>& rhs)
{
	re -= rhs.re; im -= rhs.im;
	return *this;
}

template <typename T> inline complex<T>& complex<T>::operator*=(const complex<T>& rhs)
{
	T temp_re = re * rhs.re - im * rhs.im;
	im = re * rhs.im + im * rhs.re;
	re = temp_re;
	return *this;
}

template <typename T> inline complex<T>& complex<T>::operator/=(const complex<T>& rhs)
{
	T denom = rhs.re * rhs.re + rhs.im * rhs.im;
	T temp_re = (re * rhs.re + im * rhs.im) / denom;
	im = (im * rhs.re - re * rhs.im) / denom;
	re = temp_re;
	return *this;
}

template <typename T> inline complex<T>& complex<T>::operator+=(const T& rhs)
{
	re += rhs; 
	return *this;
}

template <typename T> inline complex<T>& complex<T>::operator-=(const T& rhs)
{
	re -= rhs; 
	return *this;
}

template <typename T> inline complex<T>& complex<T>::operator*=(const T& rhs)
{
	re *= rhs; im *= rhs; 
	return *this;
}

template <typename T> inline complex<T>& complex<T>::operator/=(const T& rhs)
{
	re /= rhs; im /= rhs; 
	return *this;
}

template <typename T> inline bool isnan(const complex<T>& c)
{
	return isnan(c.re) || isnan(c.im);
}

}
