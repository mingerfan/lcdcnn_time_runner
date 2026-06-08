package ckks

import "fmt"

func mustCT(op string, x *CT) {
	if x == nil {
		panic(op + ": ct is nil")
	}
}

func normalizeRotation(step int) int {
	step %= Slots
	if step < 0 {
		step += Slots
	}
	return step
}

// Rotate returns a copy of x with slots rotated left by step.
// Negative step values rotate to the right.
func Rotate(x CT, step int) CT {
	step = normalizeRotation(step)
	if step == 0 {
		return x
	}

	var out CT
	for i := 0; i < Slots; i++ {
		out.data[i] = x.data[(i+step)%Slots]
	}
	return out
}

// RotateInPlace rotates every ct in xs left by step.
// Negative step values rotate to the right.
func RotateInPlace(xs []CT, step int) {
	step = normalizeRotation(step)
	if step == 0 {
		return
	}

	for i := range xs {
		xs[i] = Rotate(xs[i], step)
	}
}

func rotate(xs []ct, step int32) {
	RotateInPlace(xs, int(step))
}

func binaryOp(op string, a, b *CT, f func(float64, float64) float64) CT {
	mustCT(op, a)
	mustCT(op, b)

	var out CT
	for i := 0; i < Slots; i++ {
		out.data[i] = f(a.data[i], b.data[i])
	}
	return out
}

func binaryOpInPlace(op string, dst, src *CT, f func(float64, float64) float64) {
	mustCT(op, dst)
	mustCT(op, src)

	for i := 0; i < Slots; i++ {
		dst.data[i] = f(dst.data[i], src.data[i])
	}
}

func unaryOp(op string, x *CT, f func(float64) float64) CT {
	mustCT(op, x)

	var out CT
	for i := 0; i < Slots; i++ {
		out.data[i] = f(x.data[i])
	}
	return out
}

func unaryOpInPlace(op string, x *CT, f func(float64) float64) {
	mustCT(op, x)

	for i := 0; i < Slots; i++ {
		x.data[i] = f(x.data[i])
	}
}

func Add(a, b *CT) CT {
	return binaryOp("Add", a, b, func(x, y float64) float64 {
		return x + y
	})
}

func AddInPlace(dst, src *CT) {
	binaryOpInPlace("AddInPlace", dst, src, func(x, y float64) float64 {
		return x + y
	})
}

func AddScalar(x *CT, scalar float64) CT {
	return unaryOp("AddScalar", x, func(v float64) float64 {
		return v + scalar
	})
}

func AddScalarInPlace(x *CT, scalar float64) {
	unaryOpInPlace("AddScalarInPlace", x, func(v float64) float64 {
		return v + scalar
	})
}

func Sub(a, b *CT) CT {
	return binaryOp("Sub", a, b, func(x, y float64) float64 {
		return x - y
	})
}

func SubInPlace(dst, src *CT) {
	binaryOpInPlace("SubInPlace", dst, src, func(x, y float64) float64 {
		return x - y
	})
}

func Neg(x *CT) CT {
	return unaryOp("Neg", x, func(v float64) float64 {
		return -v
	})
}

func NegInPlace(x *CT) {
	unaryOpInPlace("NegInPlace", x, func(v float64) float64 {
		return -v
	})
}

func Mul(a, b *CT) CT {
	return binaryOp("Mul", a, b, func(x, y float64) float64 {
		return x * y
	})
}

func MulInPlace(dst, src *CT) {
	binaryOpInPlace("MulInPlace", dst, src, func(x, y float64) float64 {
		return x * y
	})
}

func MulScalar(x *CT, scalar float64) CT {
	return unaryOp("MulScalar", x, func(v float64) float64 {
		return v * scalar
	})
}

func MulScalarInPlace(x *CT, scalar float64) {
	unaryOpInPlace("MulScalarInPlace", x, func(v float64) float64 {
		return v * scalar
	})
}

func Square(x *CT) CT {
	return unaryOp("Square", x, func(v float64) float64 {
		return v * v
	})
}

func SquareInPlace(x *CT) {
	unaryOpInPlace("SquareInPlace", x, func(v float64) float64 {
		return v * v
	})
}

func Rescale(x CT) CT {
	return x
}

func RescaleInPlace(_ *CT) {}

func Relinearize(x CT) CT {
	return x
}

func RelinearizeInPlace(_ *CT) {}

func Bootstrap(x CT) CT {
	return x
}

func BootstrapInPlace(_ *CT) {}

// PolyEval evaluates coeffs[0] + coeffs[1]*x + ... slot-wise.
func PolyEval(x *CT, coeffs []float64) CT {
	mustCT("PolyEval", x)

	if len(coeffs) == 0 {
		return CT{}
	}

	var out CT
	for i, v := range x.data {
		acc := coeffs[len(coeffs)-1]
		for j := len(coeffs) - 2; j >= 0; j-- {
			acc = acc*v + coeffs[j]
		}
		out.data[i] = acc
	}
	return out
}

func PolyEvalInPlace(x *CT, coeffs []float64) {
	mustCT("PolyEvalInPlace", x)

	if len(coeffs) == 0 {
		*x = CT{}
		return
	}

	for i, v := range x.data {
		acc := coeffs[len(coeffs)-1]
		for j := len(coeffs) - 2; j >= 0; j-- {
			acc = acc*v + coeffs[j]
		}
		x.data[i] = acc
	}
}

// SumSlots returns a ct where every slot contains the sum of all slots in x.
func SumSlots(x *CT) CT {
	mustCT("SumSlots", x)

	sum := 0.0
	for _, v := range x.data {
		sum += v
	}
	return NewFilledCT(sum)
}

// RotateAndAdd returns x + Rotate(x, batch) + ... + Rotate(x, batch*(n-1)).
func RotateAndAdd(x *CT, batch, n int) CT {
	mustCT("RotateAndAdd", x)
	if batch <= 0 {
		panic(fmt.Sprintf("RotateAndAdd: batch must be positive, got %d", batch))
	}
	if n <= 0 {
		panic(fmt.Sprintf("RotateAndAdd: n must be positive, got %d", n))
	}

	out := *x
	for i := 1; i < n; i++ {
		rotated := Rotate(*x, i*batch)
		AddInPlace(&out, &rotated)
	}
	return out
}

// InnerSum divides the slots into groups of batch*n values. For each group it
// adds n sub-vectors of length batch and stores the result in the leftmost
// sub-vector. Other slots keep the same garbage values produced by RotateAndAdd.
func InnerSum(x *CT, batch, n int) CT {
	mustCT("InnerSum", x)
	if batch <= 0 {
		panic(fmt.Sprintf("InnerSum: batch must be positive, got %d", batch))
	}
	if n <= 0 {
		panic(fmt.Sprintf("InnerSum: n must be positive, got %d", n))
	}

	groupSize := batch * n
	if groupSize > Slots {
		panic(fmt.Sprintf("InnerSum: batch*n exceeds slots: %d*%d > %d", batch, n, Slots))
	}
	if Slots%groupSize != 0 {
		panic(fmt.Sprintf("InnerSum: batch*n must divide slots: %d %% %d != 0", Slots, groupSize))
	}

	return RotateAndAdd(x, batch, n)
}

// Replicate is the inverse-shaped companion of InnerSum. It repeatedly adds
// right rotations by batch slots, which replicates a sub-vector when the gap
// between sub-vectors is zero-filled.
func Replicate(x *CT, batch, n int) CT {
	mustCT("Replicate", x)
	if batch <= 0 {
		panic(fmt.Sprintf("Replicate: batch must be positive, got %d", batch))
	}
	if n <= 0 {
		panic(fmt.Sprintf("Replicate: n must be positive, got %d", n))
	}

	out := *x
	for i := 1; i < n; i++ {
		rotated := Rotate(*x, -i*batch)
		AddInPlace(&out, &rotated)
	}
	return out
}

func add(a *ct, b *ct) ct {
	return Add(a, b)
}
