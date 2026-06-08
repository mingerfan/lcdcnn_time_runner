package refimplement

import (
	"math"
	"testing"
)

func TestConv2DStridePadding(t *testing.T) {
	x := Tensor3DFromData(1, 3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	w := Kernel4DFromData(1, 1, 3, 3, []float64{
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
	})

	got := Conv2D(x, w, nil, Conv2DOptions{
		StrideH: 2, StrideW: 2,
		PadTop: 1, PadBottom: 1,
		PadLeft: 1, PadRight: 1,
	})
	want := Tensor3DFromData(1, 2, 2, []float64{
		12, 16,
		24, 28,
	})

	assertTensorClose(t, got, want, 0)
}

func TestConv2DGroups(t *testing.T) {
	x := Tensor3DFromData(2, 1, 1, []float64{2, 3})
	w := Kernel4DFromData(2, 1, 1, 1, []float64{10, 100})

	got := Conv2D(x, w, []float64{1, 2}, Conv2DOptions{Groups: 2})
	want := Tensor3DFromData(2, 1, 1, []float64{21, 302})

	assertTensorClose(t, got, want, 0)
}

func TestConv2DDilation(t *testing.T) {
	x := Tensor3DFromData(1, 3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	w := Kernel4DFromData(1, 1, 2, 2, []float64{
		1, 10,
		100, 1000,
	})

	got := Conv2D(x, w, nil, Conv2DOptions{DilationH: 2, DilationW: 2})
	want := Tensor3DFromData(1, 1, 1, []float64{1 + 30 + 700 + 9000})

	assertTensorClose(t, got, want, 0)
}

func TestAvgPool2DExcludePad(t *testing.T) {
	x := Tensor3DFromData(1, 2, 2, []float64{
		1, 2,
		3, 4,
	})

	got := AvgPool2D(x, Pool2DOptions{
		KernelH: 2, KernelW: 2,
		StrideH: 1, StrideW: 1,
		PadTop: 1, PadBottom: 1,
		PadLeft: 1, PadRight: 1,
	})
	want := Tensor3DFromData(1, 3, 3, []float64{
		1, 1.5, 2,
		2, 2.5, 3,
		3, 3.5, 4,
	})

	assertTensorClose(t, got, want, 1e-12)
}

func TestAvgPool2DIncludePad(t *testing.T) {
	x := Tensor3DFromData(1, 2, 2, []float64{
		1, 2,
		3, 4,
	})

	got := AvgPool2D(x, Pool2DOptions{
		KernelH: 2, KernelW: 2,
		StrideH: 1, StrideW: 1,
		PadTop: 1, PadBottom: 1,
		PadLeft: 1, PadRight: 1,
		CountIncludePad: true,
	})
	want := Tensor3DFromData(1, 3, 3, []float64{
		0.25, 0.75, 0.5,
		1, 2.5, 1.5,
		0.75, 1.75, 1,
	})

	assertTensorClose(t, got, want, 1e-12)
}

func TestMaxPool2DUsesNegativeInfinityPadding(t *testing.T) {
	x := Tensor3DFromData(1, 2, 2, []float64{
		1, 2,
		3, 4,
	})

	got := MaxPool2D(x, Pool2DOptions{
		KernelH: 2, KernelW: 2,
		StrideH: 1, StrideW: 1,
		PadTop: 1, PadBottom: 1,
		PadLeft: 1, PadRight: 1,
	})
	want := Tensor3DFromData(1, 3, 3, []float64{
		1, 2, 2,
		3, 4, 4,
		3, 4, 4,
	})

	assertTensorClose(t, got, want, 0)
}

func TestGlobalAvgPool2D(t *testing.T) {
	x := Tensor3DFromData(2, 2, 2, []float64{
		1, 2,
		3, 4,
		10, 20,
		30, 40,
	})

	got := GlobalAvgPool2D(x)
	want := Tensor3DFromData(2, 1, 1, []float64{2.5, 25})

	assertTensorClose(t, got, want, 0)
}

func TestActivationsAndPolyEval(t *testing.T) {
	x := Tensor3DFromData(1, 1, 3, []float64{-1, 0, 2})

	relu := ReLU(x)
	assertTensorClose(t, relu, Tensor3DFromData(1, 1, 3, []float64{0, 0, 2}), 0)

	swish := Swish(x, 1)
	assertTensorClose(t, swish, Tensor3DFromData(1, 1, 3, []float64{
		-1 / (1 + math.Exp(1)),
		0,
		2 / (1 + math.Exp(-2)),
	}), 1e-12)

	poly := PolyEval(x, []float64{1, 2, 3})
	assertTensorClose(t, poly, Tensor3DFromData(1, 1, 3, []float64{2, 1, 17}), 1e-12)
}

func TestAddAndAddInPlace(t *testing.T) {
	a := Tensor3DFromData(1, 1, 3, []float64{1, 2, 3})
	b := Tensor3DFromData(1, 1, 3, []float64{10, 20, 30})

	got := Add(a, b)
	assertTensorClose(t, got, Tensor3DFromData(1, 1, 3, []float64{11, 22, 33}), 0)

	AddInPlace(a, b)
	assertTensorClose(t, a, Tensor3DFromData(1, 1, 3, []float64{11, 22, 33}), 0)
}

func assertTensorClose(t *testing.T, got, want Tensor3D, tol float64) {
	t.Helper()
	got.Validate()
	want.Validate()

	if got.C != want.C || got.H != want.H || got.W != want.W {
		t.Fatalf("shape mismatch: got (%d,%d,%d), want (%d,%d,%d)", got.C, got.H, got.W, want.C, want.H, want.W)
	}
	for i := range got.Data {
		diff := math.Abs(got.Data[i] - want.Data[i])
		if diff > tol {
			t.Fatalf("data[%d] mismatch: got %.17g, want %.17g, diff %.17g > %.17g", i, got.Data[i], want.Data[i], diff, tol)
		}
	}
}
