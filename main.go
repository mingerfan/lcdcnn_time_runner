package main

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func main() {
	paramsLiteral := ckks.ParametersLiteral{
		LogN: 10,
		LogQ: []int{45, 45},
		LogP: []int{45},
	}

	params, err := ckks.NewParametersFromLiteral(paramsLiteral)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Lattigo CKKS params ready: LogN=%d, LogSlots=%d\n", params.LogN(), params.LogMaxSlots())
}
