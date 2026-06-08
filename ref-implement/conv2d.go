package refimplement

import "fmt"

type Conv2DOptions struct {
	StrideH, StrideW     int
	PadTop, PadBottom    int
	PadLeft, PadRight    int
	DilationH, DilationW int
	Groups               int
}

// Conv2D computes deep-learning-style cross-correlation, not kernel-flipped convolution.
func Conv2D(x Tensor3D, w Kernel4D, bias []float64, opts Conv2DOptions) Tensor3D {
	x.Validate()
	w.Validate()
	opts = normalizeConv2DOptions(opts)

	if x.C%opts.Groups != 0 {
		panic(fmt.Sprintf("input channels must be divisible by groups: C=%d, groups=%d", x.C, opts.Groups))
	}
	if w.Cout%opts.Groups != 0 {
		panic(fmt.Sprintf("output channels must be divisible by groups: Cout=%d, groups=%d", w.Cout, opts.Groups))
	}

	cinPerGroup := x.C / opts.Groups
	coutPerGroup := w.Cout / opts.Groups
	if w.Cin != cinPerGroup {
		panic(fmt.Sprintf("kernel Cin must equal input Cin/groups: got %d, want %d", w.Cin, cinPerGroup))
	}
	if bias != nil && len(bias) != w.Cout {
		panic(fmt.Sprintf("bias length mismatch: got %d, want %d", len(bias), w.Cout))
	}

	outH := spatialOutDim(x.H, w.Kh, opts.StrideH, opts.PadTop, opts.PadBottom, opts.DilationH)
	outW := spatialOutDim(x.W, w.Kw, opts.StrideW, opts.PadLeft, opts.PadRight, opts.DilationW)
	if outH <= 0 || outW <= 0 {
		panic(fmt.Sprintf("invalid Conv2D output shape: (%d,%d)", outH, outW))
	}

	out := NewTensor3D(w.Cout, outH, outW)
	for co := 0; co < w.Cout; co++ {
		group := co / coutPerGroup
		ciStart := group * cinPerGroup

		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				sum := 0.0
				if bias != nil {
					sum = bias[co]
				}

				for ciLocal := 0; ciLocal < cinPerGroup; ciLocal++ {
					ci := ciStart + ciLocal
					for kh := 0; kh < w.Kh; kh++ {
						ih := oh*opts.StrideH + kh*opts.DilationH - opts.PadTop
						if ih < 0 || ih >= x.H {
							continue
						}

						for kw := 0; kw < w.Kw; kw++ {
							iw := ow*opts.StrideW + kw*opts.DilationW - opts.PadLeft
							if iw < 0 || iw >= x.W {
								continue
							}

							xv := x.Data[index3(ci, ih, iw, x.H, x.W)]
							wv := w.Data[w.indexUnchecked(co, ciLocal, kh, kw)]
							sum += xv * wv
						}
					}
				}

				out.Data[index3(co, oh, ow, outH, outW)] = sum
			}
		}
	}
	return out
}

func normalizeConv2DOptions(opts Conv2DOptions) Conv2DOptions {
	if opts.StrideH == 0 {
		opts.StrideH = 1
	}
	if opts.StrideW == 0 {
		opts.StrideW = 1
	}
	if opts.DilationH == 0 {
		opts.DilationH = 1
	}
	if opts.DilationW == 0 {
		opts.DilationW = 1
	}
	if opts.Groups == 0 {
		opts.Groups = 1
	}

	mustPositive("StrideH", opts.StrideH)
	mustPositive("StrideW", opts.StrideW)
	mustPositive("DilationH", opts.DilationH)
	mustPositive("DilationW", opts.DilationW)
	mustPositive("Groups", opts.Groups)
	mustNonNegative("PadTop", opts.PadTop)
	mustNonNegative("PadBottom", opts.PadBottom)
	mustNonNegative("PadLeft", opts.PadLeft)
	mustNonNegative("PadRight", opts.PadRight)
	return opts
}

func spatialOutDim(in, kernel, stride, padBefore, padAfter, dilation int) int {
	effectiveKernel := dilation*(kernel-1) + 1
	available := in + padBefore + padAfter - effectiveKernel
	if available < 0 {
		return 0
	}
	return available/stride + 1
}
