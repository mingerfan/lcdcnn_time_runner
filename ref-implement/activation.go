package refimplement

import "math"

// ReLU applies max(0, x) element-wise.
func ReLU(x Tensor3D) Tensor3D {
	x.Validate()

	out := NewTensor3D(x.C, x.H, x.W)
	for i, v := range x.Data {
		if v > 0 {
			out.Data[i] = v
		}
	}
	return out
}

// ReLUInPlace applies max(0, x) element-wise and modifies x.
func ReLUInPlace(x Tensor3D) {
	x.Validate()

	for i, v := range x.Data {
		if v < 0 {
			x.Data[i] = 0
		}
	}
}

// Swish applies x * sigmoid(beta*x) element-wise.
func Swish(x Tensor3D, beta float64) Tensor3D {
	x.Validate()

	out := NewTensor3D(x.C, x.H, x.W)
	for i, v := range x.Data {
		out.Data[i] = v * sigmoid(beta*v)
	}
	return out
}

// SwishInPlace applies x * sigmoid(beta*x) element-wise and modifies x.
func SwishInPlace(x Tensor3D, beta float64) {
	x.Validate()

	for i, v := range x.Data {
		x.Data[i] = v * sigmoid(beta*v)
	}
}

// PolyEval evaluates coeffs[0] + coeffs[1]*x + ... element-wise.
func PolyEval(x Tensor3D, coeffs []float64) Tensor3D {
	x.Validate()

	out := NewTensor3D(x.C, x.H, x.W)
	for i, v := range x.Data {
		out.Data[i] = PolyEvalScalar(v, coeffs)
	}
	return out
}

// PolyEvalScalar evaluates coeffs[0] + coeffs[1]*x + ... with Horner's method.
func PolyEvalScalar(x float64, coeffs []float64) float64 {
	if len(coeffs) == 0 {
		return 0
	}

	acc := coeffs[len(coeffs)-1]
	for i := len(coeffs) - 2; i >= 0; i-- {
		acc = acc*x + coeffs[i]
	}
	return acc
}

func sigmoid(x float64) float64 {
	if x >= 0 {
		z := math.Exp(-x)
		return 1 / (1 + z)
	}

	z := math.Exp(x)
	return z / (1 + z)
}
