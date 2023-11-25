from memory import memset_zero, memset
from algorithm import vectorize, parallelize
from math import mod, trunc


alias nelts = 16 * simdwidthof[DType.int8]()


struct Matrix[dtype: DType = DType.int8]:
    var dim0: Int
    var dim1: Int
    var dim2: Int
    var _data: DTypePointer[dtype]
    alias simd_width: Int = simdwidthof[dtype]()

    fn __init__(inout self, *dims: Int):
        self.dim0 = dims[0]
        self.dim1 = dims[1]
        self.dim2 = 1

        if len(dims) == 3:
            self.dim2 = dims[2]
        self._data = DTypePointer[dtype].alloc(self.dim0 * self.dim1 * self.dim2)
        memset_zero[dtype](self._data, self.dim0 * self.dim1 * self.dim2)

    fn __copyinit__(inout self, other: Self):
        self._data = other._data
        self.dim0 = other.dim0
        self.dim1 = other.dim1
        self.dim2 = other.dim2

    fn __getitem__(self, x: Int, y: Int) -> SIMD[dtype, 1]:
        return self._data.simd_load[1](x * self.dim1 + y)

    fn __setitem__(inout self, x: Int, y: Int, value: SIMD[dtype, 1]):
        self._data.simd_store[1](x * self.dim1 + y, value)

    fn __getitem__(self, x: Int, y: Int, z: Int) -> SIMD[dtype, 1]:
        return self._data.simd_load[1](z * self.dim2 + x * self.dim1 + y)

    fn __setitem__(inout self, x: Int, y: Int, z: Int, value: SIMD[dtype, 1]):
        self._data.simd_store[1](z * self.dim2 + x * self.dim1 + y, value)

    fn fill(inout self, value: SIMD[dtype, 1]):
        let cnt = self.dim0 * self.dim1 * self.dim2
        var tmp = SIMD[dtype, nelts](value)

        @parameter
        fn _fill[_nelts: Int](j: Int):
            self._data.simd_store[_nelts](j, value)

        vectorize[nelts, _fill](cnt)

    fn print(self, z: Int = 0) -> None:
        var rank: Int = 2
        var dim0: Int = 0
        var dim1: Int = 0
        var val: SIMD[dtype, 1] = 0.0
        let prec: Int = 2
        if self.dim0 == 1:
            rank = 1
            dim0 = 1
            dim1 = self.dim1
        else:
            dim0 = self.dim0
            dim1 = self.dim1

        if dim0 > 0 and dim1 > 0:
            for j in range(dim0):
                if rank > 1:
                    if j == 0:
                        print_no_newline("  [")
                    else:
                        print_no_newline("\n   ")
                print_no_newline("[")
                for k in range(dim1):
                    if rank == 1:
                        val = self._data.simd_load[1](k)
                    if rank == 2:
                        val = self[j, k, z]

                    var cval = val.cast[DType.float32]()
                    let int_str = String(trunc(cval).cast[DType.int32]())
                    let float_str: String
                    float_str = String(mod(cval,1))
                    let s = int_str+"."+float_str[2:prec+2]

                    if k == 0:
                        print_no_newline(s)
                    else:
                        print_no_newline("  ", s)
                print_no_newline("]")
            if rank > 1:
                print_no_newline("]")
            print()
            if rank > 2:
                print("]")
