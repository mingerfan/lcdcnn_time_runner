package refimplement

import "fmt"

// Tensor3D stores one image or feature map in CHW layout.
type Tensor3D struct {
	C, H, W int
	Data    []float64
}

// Kernel4D stores convolution weights as [Cout, CinPerGroup, Kh, Kw].
type Kernel4D struct {
	Cout, Cin, Kh, Kw int
	Data              []float64
}

func NewTensor3D(c, h, w int) Tensor3D {
	mustPositive("C", c)
	mustPositive("H", h)
	mustPositive("W", w)

	return Tensor3D{
		C:    c,
		H:    h,
		W:    w,
		Data: make([]float64, c*h*w),
	}
}

func Tensor3DFromData(c, h, w int, data []float64) Tensor3D {
	t := NewTensor3D(c, h, w)
	if len(data) != len(t.Data) {
		panic(fmt.Sprintf("tensor data length mismatch: got %d, want %d", len(data), len(t.Data)))
	}
	copy(t.Data, data)
	return t
}

func (t Tensor3D) Shape() (c, h, w int) {
	return t.C, t.H, t.W
}

func (t Tensor3D) Len() int {
	return t.C * t.H * t.W
}

func (t Tensor3D) Validate() {
	mustPositive("C", t.C)
	mustPositive("H", t.H)
	mustPositive("W", t.W)
	if len(t.Data) != t.Len() {
		panic(fmt.Sprintf("tensor data length mismatch: got %d, want %d", len(t.Data), t.Len()))
	}
}

func (t Tensor3D) Clone() Tensor3D {
	t.Validate()
	out := NewTensor3D(t.C, t.H, t.W)
	copy(out.Data, t.Data)
	return out
}

func (t Tensor3D) Index(c, h, w int) int {
	t.Validate()
	if c < 0 || c >= t.C || h < 0 || h >= t.H || w < 0 || w >= t.W {
		panic(fmt.Sprintf("tensor index out of range: got (%d,%d,%d), shape (%d,%d,%d)", c, h, w, t.C, t.H, t.W))
	}
	return index3(c, h, w, t.H, t.W)
}

func (t Tensor3D) At(c, h, w int) float64 {
	return t.Data[t.Index(c, h, w)]
}

func (t Tensor3D) Set(c, h, w int, value float64) {
	t.Data[t.Index(c, h, w)] = value
}

func NewKernel4D(cout, cin, kh, kw int) Kernel4D {
	mustPositive("Cout", cout)
	mustPositive("Cin", cin)
	mustPositive("Kh", kh)
	mustPositive("Kw", kw)

	return Kernel4D{
		Cout: cout,
		Cin:  cin,
		Kh:   kh,
		Kw:   kw,
		Data: make([]float64, cout*cin*kh*kw),
	}
}

func Kernel4DFromData(cout, cin, kh, kw int, data []float64) Kernel4D {
	k := NewKernel4D(cout, cin, kh, kw)
	if len(data) != len(k.Data) {
		panic(fmt.Sprintf("kernel data length mismatch: got %d, want %d", len(data), len(k.Data)))
	}
	copy(k.Data, data)
	return k
}

func (k Kernel4D) Shape() (cout, cin, kh, kw int) {
	return k.Cout, k.Cin, k.Kh, k.Kw
}

func (k Kernel4D) Len() int {
	return k.Cout * k.Cin * k.Kh * k.Kw
}

func (k Kernel4D) Validate() {
	mustPositive("Cout", k.Cout)
	mustPositive("Cin", k.Cin)
	mustPositive("Kh", k.Kh)
	mustPositive("Kw", k.Kw)
	if len(k.Data) != k.Len() {
		panic(fmt.Sprintf("kernel data length mismatch: got %d, want %d", len(k.Data), k.Len()))
	}
}

func (k Kernel4D) Clone() Kernel4D {
	k.Validate()
	out := NewKernel4D(k.Cout, k.Cin, k.Kh, k.Kw)
	copy(out.Data, k.Data)
	return out
}

func (k Kernel4D) Index(co, ci, kh, kw int) int {
	k.Validate()
	if co < 0 || co >= k.Cout || ci < 0 || ci >= k.Cin || kh < 0 || kh >= k.Kh || kw < 0 || kw >= k.Kw {
		panic(fmt.Sprintf(
			"kernel index out of range: got (%d,%d,%d,%d), shape (%d,%d,%d,%d)",
			co, ci, kh, kw, k.Cout, k.Cin, k.Kh, k.Kw,
		))
	}
	return k.indexUnchecked(co, ci, kh, kw)
}

func (k Kernel4D) At(co, ci, kh, kw int) float64 {
	return k.Data[k.Index(co, ci, kh, kw)]
}

func (k Kernel4D) Set(co, ci, kh, kw int, value float64) {
	k.Data[k.Index(co, ci, kh, kw)] = value
}

func (k Kernel4D) indexUnchecked(co, ci, kh, kw int) int {
	return (((co*k.Cin)+ci)*k.Kh+kh)*k.Kw + kw
}

func index3(c, h, w, hSize, wSize int) int {
	return (c*hSize+h)*wSize + w
}

func mustPositive(name string, value int) {
	if value <= 0 {
		panic(fmt.Sprintf("%s must be positive, got %d", name, value))
	}
}

func mustNonNegative(name string, value int) {
	if value < 0 {
		panic(fmt.Sprintf("%s must be non-negative, got %d", name, value))
	}
}

func assertSameShape(op string, a, b Tensor3D) {
	a.Validate()
	b.Validate()
	if a.C != b.C || a.H != b.H || a.W != b.W {
		panic(fmt.Sprintf(
			"%s shape mismatch: got (%d,%d,%d) and (%d,%d,%d)",
			op, a.C, a.H, a.W, b.C, b.H, b.W,
		))
	}
}
