use ndarray::Array;
use ndarray_rand::RandomExt;
use num_traits::Float;
use teenygrad::helpers::{DType, dtypes, DEBUG};
use teenygrad::ops::{UnaryOps, BinaryOps, ReduceOps, TernaryOps, LoadOps};

pub struct RawCPUBuffer<T> {
    x: T,
}

impl <T> RawCPUBuffer<T> {
    pub fn to_cpu(self) -> T {
        self.x
    }
}

pub struct LazyBuffer<A, D> {
    _np: Array<A, D>,
}

impl<A, D> LazyBuffer<A, D> {
    pub fn base(&self) -> &Self {
        self
    }

    pub fn dtype(&self) -> DType {
        dtypes::from_np(self._np.dtype())
    }

    pub fn realized(&self) -> RawCPUBuffer<Array<A, D>> {
        RawCPUBuffer { x: self._np.clone() }
    }

    pub fn shape(&self) -> D {
        self._np.shape()
    }

    pub fn schedule(&self, seen: Option<&Seen>) -> Vec<ScheduleItem> {
        vec![]
    }

    pub fn is_unrealized_contiguous_const(&self) -> bool {
        false
    }

    pub fn copy_to_device(&self, device: &str) -> LazyBuffer {
        self.clone()
    }

    pub fn from_cpu(x: Array<A, D>) -> LazyBuffer<A, D> {
        LazyBuffer { _np: x}
    }

    pub fn loadop(op: LoadOps, shape: D, dtype: DType, device: &str, arg: Option<f64>, src: Option<LazyBuffer<A, D>>) -> LazyBuffer<A, D> { 
        match op {
            LoadOps::RAND => LazyBuffer {_np: Array::random(shape, dtype.np) },
            LoadOps::CONST => LazyBuffer {_np: Array::from_elem(shape, arg.unwrap()) },
            LoadOps::EMPTY => LazyBuffer { _np: Array::default(shape) },
        }
    }

    pub fn contiguous(&self, x: Array<A, D>) -> Array<A, D> {
        x
    }

    pub fn constant(&self, x: A) -> LazyBuffer<A, D> {
        LazyBuffer { _np: Array::from_elem(self._np.raw_dim(), x) }
    }

    pub fn e(&self, op: UnaryOps, srcs: &[LazyBuffer<A, D>]) -> LazyBuffer<A, D> {
        match op {
            UnaryOps::NEG => LazyBuffer { _np: -&self._np },
            UnaryOps::EXP2 => LazyBuffer { _np: self._np.mapv(Float::exp2) },
            UnaryOps::LOG2 => LazyBuffer { _np: self._np.mapv(Float::log2) },
            UnaryOps::SIN => LazyBuffer { _np: self._np.mapv(Float::sin) },
            UnaryOps::SQRT => LazyBuffer { _np: self._np.mapv(Float::sqrt) },
            UnaryOps::ADD => LazyBuffer { _np: &self._np + &srcs[0]._np },
            UnaryOps::SUB => LazyBuffer { _np: &self._np - &srcs[0]._np },
            UnaryOps::MUL => LazyBuffer { _np: &self._np * &srcs[0]._np },
            UnaryOps::DIV => LazyBuffer { _np: &self._np / &srcs[0]._np },
            UnaryOps::MAX => LazyBuffer { _np: self._np.mapv(|x| x.max(srcs[0]._np)) },
            UnaryOps::CMPLT => LazyBuffer { _np: self._np.lt(&srcs[0]._np) },
            UnaryOps::WHERE => LazyBuffer { _np: self._np.mapv(|x| if x { srcs[0]._np } else { srcs[1]._np }) },
        }
    }

    pub fn r(&self, op: ReduceOps, new_shape: D) -> LazyBuffer<A, D> {
        match op {
            ReduceOps::SUM => LazyBuffer { _np: self._np.sum_axis(Axis(0)).insert_axis(Axis(0)) }, 
            ReduceOps::MAX => LazyBuffer { _np: self._np.fold_axis(Axis(0), A::min_value, |&a, &b| a.max, A::min_value, |&a, &b| a.max(b)) } 
        }
    }

    pub fn reshape(&self, arg: D) -> LazyBuffer<A, D> {
        LazyBuffer {_np: self._np.into_shape(arg).unwrap() }
    }

    pub fn expand(&self, arg: D) -> LazyBuffer<A, D> {
        LazyBuffer { _np: self._np.broadcast(arg).unwrap() }
    }

    pub fn shrink(&self, arg: &[(usize, usize)]) -> LazyBuffer<A, D> {
        LazyBuffer { _np: self._np.slice(&[arg[0].clone(), arg[1].clone()]).to_owned() }
    }

    pub fn permute(&self, arg: &[usize]) -> LazyBuffer<A, D> {
        LazyBuffer { _np: self._np.permuted_axes(arg) }
    }

    pub fn pad(&self, arg: &[(usize, usize)]) -> LazyBuffer<A, D> {
        LazyBuffer {_np: self._np.pad(arg, 0) }
    }

    pub fn stride(&self, arg: &[usize]) -> LazyBuffer<A, D> {
        LazyBuffer { _np: self._np.slice_move(arg) }
    }
}