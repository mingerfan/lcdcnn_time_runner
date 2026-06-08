package refimplement

import (
	"fmt"
	"math"
)

type Pool2DOptions struct {
	KernelH, KernelW     int
	StrideH, StrideW     int
	PadTop, PadBottom    int
	PadLeft, PadRight    int
	DilationH, DilationW int
	CountIncludePad      bool
}

// AvgPool2D averages each channel independently.
func AvgPool2D(x Tensor3D, opts Pool2DOptions) Tensor3D {
	x.Validate()
	opts = normalizePool2DOptions(opts)

	outH := spatialOutDim(x.H, opts.KernelH, opts.StrideH, opts.PadTop, opts.PadBottom, opts.DilationH)
	outW := spatialOutDim(x.W, opts.KernelW, opts.StrideW, opts.PadLeft, opts.PadRight, opts.DilationW)
	if outH <= 0 || outW <= 0 {
		panic(fmt.Sprintf("invalid AvgPool2D output shape: (%d,%d)", outH, outW))
	}

	out := NewTensor3D(x.C, outH, outW)
	for c := 0; c < x.C; c++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				sum := 0.0
				validCount := 0

				for kh := 0; kh < opts.KernelH; kh++ {
					ih := oh*opts.StrideH + kh*opts.DilationH - opts.PadTop
					for kw := 0; kw < opts.KernelW; kw++ {
						iw := ow*opts.StrideW + kw*opts.DilationW - opts.PadLeft
						if ih < 0 || ih >= x.H || iw < 0 || iw >= x.W {
							continue
						}

						sum += x.Data[index3(c, ih, iw, x.H, x.W)]
						validCount++
					}
				}

				denom := validCount
				if opts.CountIncludePad {
					denom = opts.KernelH * opts.KernelW
				}
				if denom == 0 {
					panic("AvgPool2D window has no valid input values")
				}
				out.Data[index3(c, oh, ow, outH, outW)] = sum / float64(denom)
			}
		}
	}
	return out
}

// MaxPool2D maxes each channel independently. Padding values are treated as -Inf.
func MaxPool2D(x Tensor3D, opts Pool2DOptions) Tensor3D {
	x.Validate()
	opts = normalizePool2DOptions(opts)

	outH := spatialOutDim(x.H, opts.KernelH, opts.StrideH, opts.PadTop, opts.PadBottom, opts.DilationH)
	outW := spatialOutDim(x.W, opts.KernelW, opts.StrideW, opts.PadLeft, opts.PadRight, opts.DilationW)
	if outH <= 0 || outW <= 0 {
		panic(fmt.Sprintf("invalid MaxPool2D output shape: (%d,%d)", outH, outW))
	}

	out := NewTensor3D(x.C, outH, outW)
	for c := 0; c < x.C; c++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				maxValue := math.Inf(-1)
				seen := false

				for kh := 0; kh < opts.KernelH; kh++ {
					ih := oh*opts.StrideH + kh*opts.DilationH - opts.PadTop
					for kw := 0; kw < opts.KernelW; kw++ {
						iw := ow*opts.StrideW + kw*opts.DilationW - opts.PadLeft
						if ih < 0 || ih >= x.H || iw < 0 || iw >= x.W {
							continue
						}

						v := x.Data[index3(c, ih, iw, x.H, x.W)]
						if !seen || v > maxValue {
							maxValue = v
							seen = true
						}
					}
				}

				if !seen {
					panic("MaxPool2D window has no valid input values")
				}
				out.Data[index3(c, oh, ow, outH, outW)] = maxValue
			}
		}
	}
	return out
}

func GlobalAvgPool2D(x Tensor3D) Tensor3D {
	x.Validate()

	out := NewTensor3D(x.C, 1, 1)
	denom := float64(x.H * x.W)
	for c := 0; c < x.C; c++ {
		sum := 0.0
		for h := 0; h < x.H; h++ {
			for w := 0; w < x.W; w++ {
				sum += x.Data[index3(c, h, w, x.H, x.W)]
			}
		}
		out.Data[c] = sum / denom
	}
	return out
}

func normalizePool2DOptions(opts Pool2DOptions) Pool2DOptions {
	mustPositive("KernelH", opts.KernelH)
	mustPositive("KernelW", opts.KernelW)

	if opts.StrideH == 0 {
		opts.StrideH = opts.KernelH
	}
	if opts.StrideW == 0 {
		opts.StrideW = opts.KernelW
	}
	if opts.DilationH == 0 {
		opts.DilationH = 1
	}
	if opts.DilationW == 0 {
		opts.DilationW = 1
	}

	mustPositive("StrideH", opts.StrideH)
	mustPositive("StrideW", opts.StrideW)
	mustPositive("DilationH", opts.DilationH)
	mustPositive("DilationW", opts.DilationW)
	mustNonNegative("PadTop", opts.PadTop)
	mustNonNegative("PadBottom", opts.PadBottom)
	mustNonNegative("PadLeft", opts.PadLeft)
	mustNonNegative("PadRight", opts.PadRight)
	return opts
}
