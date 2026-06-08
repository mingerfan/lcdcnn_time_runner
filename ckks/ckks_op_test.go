package ckks

import "testing"

func TestRotateDirections(t *testing.T) {
	x := NewCT([]float64{1, 2, 3, 4})

	left := Rotate(x, 1)
	assertSlots(t, left, []float64{2, 3, 4, 0, 0})

	right := Rotate(x, -1)
	if right.At(0) != 0 || right.At(1) != 1 || right.At(2) != 2 || right.At(3) != 3 || right.At(4) != 4 {
		t.Fatalf("right rotation mismatch: got first slots %v", right.Data()[:5])
	}
}

func TestArithmeticOps(t *testing.T) {
	a := NewCT([]float64{1, 2, 3})
	b := NewCT([]float64{10, 20, 30})

	assertSlots(t, Add(&a, &b), []float64{11, 22, 33})
	assertSlots(t, Sub(&b, &a), []float64{9, 18, 27})
	assertSlots(t, Mul(&a, &b), []float64{10, 40, 90})
	assertSlots(t, MulScalar(&a, 2), []float64{2, 4, 6})
	assertSlots(t, AddScalar(&a, 5), []float64{6, 7, 8})
	assertSlots(t, Square(&a), []float64{1, 4, 9})
	assertSlots(t, Neg(&a), []float64{-1, -2, -3})
}

func TestInPlaceOps(t *testing.T) {
	a := NewCT([]float64{1, 2, 3})
	b := NewCT([]float64{10, 20, 30})

	AddInPlace(&a, &b)
	assertSlots(t, a, []float64{11, 22, 33})

	MulScalarInPlace(&a, 0.5)
	assertSlots(t, a, []float64{5.5, 11, 16.5})
}

func TestPolyEval(t *testing.T) {
	x := NewCT([]float64{-1, 0, 2})

	got := PolyEval(&x, []float64{1, 2, 3})
	assertSlots(t, got, []float64{2, 1, 17})

	PolyEvalInPlace(&x, []float64{1, 2, 3})
	assertSlots(t, x, []float64{2, 1, 17})
}

func TestSumSlotsAndRotateAndAdd(t *testing.T) {
	x := NewCT([]float64{1, 2, 3, 4, 5, 6})

	sum := SumSlots(&x)
	assertSlots(t, sum, []float64{21, 21, 21, 21, 21, 21})

	got := RotateAndAdd(&x, 2, 3)
	assertSlots(t, got, []float64{9, 12, 8, 10, 5, 6})
}

func TestInnerSum(t *testing.T) {
	x := NewCT([]float64{1, 2, 3, 4, 5, 6, 7, 8})

	got := InnerSum(&x, 2, 4)
	assertSlots(t, got, []float64{16, 20, 15, 18, 12, 14, 7, 8})
}

func TestNoOpCKKSLevelOps(t *testing.T) {
	x := NewCT([]float64{1, 2, 3})

	assertSlots(t, Rescale(x), []float64{1, 2, 3})
	assertSlots(t, Relinearize(x), []float64{1, 2, 3})
	assertSlots(t, Bootstrap(x), []float64{1, 2, 3})
}

func assertSlots(t *testing.T, got CT, want []float64) {
	t.Helper()

	for i, v := range want {
		if got.At(i) != v {
			t.Fatalf("slot %d mismatch: got %v, want %v", i, got.At(i), v)
		}
	}
}
