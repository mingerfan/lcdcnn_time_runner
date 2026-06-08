package ckks

import "fmt"

const Slots = 32768

// CT is a simulated CKKS slot vector.
//
// Plaintexts and ciphertexts intentionally use the same type in this package:
// the code models arithmetic over slots and ignores encryption metadata.
type CT struct {
	data [Slots]float64
}

// Plaintext and Ciphertext are aliases because this simulator only models slot
// arithmetic and intentionally ignores encryption metadata.
type Plaintext = CT
type Ciphertext = CT

// ct is kept as an internal alias for older package-local code.
type ct = CT

func NewCT(values []float64) CT {
	if len(values) > Slots {
		panic(fmt.Sprintf("NewCT: got %d values, max %d", len(values), Slots))
	}

	var out CT
	copy(out.data[:], values)
	return out
}

func NewFilledCT(value float64) CT {
	var out CT
	for i := range out.data {
		out.data[i] = value
	}
	return out
}

func (x CT) Len() int {
	return Slots
}

func (x CT) Data() []float64 {
	out := make([]float64, Slots)
	copy(out, x.data[:])
	return out
}

func (x CT) Clone() CT {
	return x
}

func (x CT) At(index int) float64 {
	checkSlotIndex("At", index)
	return x.data[index]
}

func (x *CT) Set(index int, value float64) {
	if x == nil {
		panic("Set: ct is nil")
	}
	checkSlotIndex("Set", index)
	x.data[index] = value
}

func (x CT) len() int {
	return x.Len()
}

func checkSlotIndex(op string, index int) {
	if index < 0 || index >= Slots {
		panic(fmt.Sprintf("%s: slot index out of range: got %d, want [0,%d)", op, index, Slots))
	}
}
