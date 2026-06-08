package refimplement

// Add returns the element-wise sum of two CHW tensors.
func Add(a, b Tensor3D) Tensor3D {
	assertSameShape("Add", a, b)

	out := NewTensor3D(a.C, a.H, a.W)
	for i := range out.Data {
		out.Data[i] = a.Data[i] + b.Data[i]
	}
	return out
}

// AddInPlace adds src into dst. The dst tensor is modified.
func AddInPlace(dst, src Tensor3D) {
	assertSameShape("AddInPlace", dst, src)

	for i := range dst.Data {
		dst.Data[i] += src.Data[i]
	}
}
