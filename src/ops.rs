use std::option::Option;

enum UnaryOps {
    NOOP,
    EXP2,
    LOG2,
    CAST,
    SIN,
    SQRT,
    RECIP,
    NEG,
}

enum BinaryOps {
    ADD,
    SUB,
    MUL,
    DIV,
    MAX,
    MOD,
    CMPLT,
}

enum ReduceOps {
    SUM,
    MAX,
}

enum TernaryOps {
    MULACC,
    WHERE,
}

enum MovementOps {
    RESHAPE,
    PERMUTE,
    EXPAND,
    PAD,
    SHRINK,
    STRIDE,
}

enum LoadOps{
    EMPTY,
    RAND,
    CONST,
    FROM,
    CONTIGUOUS,
    CUSTOM,
}

struct Device;

impl Device {
    const DEFAULT: &str = "CPU";

    fn canonicalize(device: Option<&str>) -> &str {
        "CPU"
    }
}