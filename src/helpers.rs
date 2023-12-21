use std::collections::HashMap;
use std::os;
use std::cmp::Ordering;

static OSX: bool = cfg(!target_os = "macos");

fn dedup<T: Eq + std::hash::Hash + Clone>(x: Vec<T>) -> Vec<T> {
    let mut set = HashSet::new();
    x.into_iter().filter(|e| set.insert(e.clone())).collect()

}

fn argfix<T>(x: T) -> T {
    x
}

fn make_pair<T: Clone>(x: T, cnt: usize) -> Vec<T> {
    vec![x.clone(); cnt]
}

fn flatten<T>(l: Vec<Vec<T>>) -> Vec<T> {
    l.into_iter().flatten()
}

fn argsort<T: Ord>(x: Vec<T>) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..x.len()).collect();
    indices.sort_by(|&i, &j | x[i].cmp(&x[j])); 
    indices
}

fn all_int(t: Vec<i32>) -> bool {
    t.iter().all(|&s| s.is_integer())
}

fn round_up(num: i32, amt: i32) -> i32 {
    (num + amt - 1) / amt * amt
}

fn getenv(key: &str, default: i32) -> i32 {
    match os::var(key) {
        Ok(val) => val.parse().unwrap_or(default),
        Err(_) => default,
    }
}

static DEBUG: i32 = getenv("DEBUG", 0);
static CI: bool = !os::var("CI").unwrap_or_default().is_empty();

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DType {
    priority: i32,
    itemsize: i32,
    name: String,
    np: Option<i32>,
    sz: i32,
}

impl DType {
    fn new (priority: i32, itemsize: i32, name: &str, np: Option<i32>, sz: i32) -> DType {
        DType {
            priority,
            itemsize,
            name: name.to_string(),
            np,
            sz,
        }
    }
}

fn is_int(x: &DType) -> bool {
    [int8, int16, int32, int64, uint8, uint16, uint32, uint64].contains(x)
}

fn is_float(x: &DType) -> bool {
    [float16, float32, float64].contains(x)
}

fn is_unsigned(x: &DType) -> bool {
    [uint8, uint16, uint32, uint64].contains(x)
}

fn from_np(x: i32) -> DType {
    DTYPES_DICT.get(&x).unwrap().clone()
}

static bool: DType = DType::new(0, 1, "bool", Some(0), 1);
static float16: DType = DType::new(9, 2, "half", Some(1), 1);
static float32: DType = DType::new(10, 4, "float", Some(2), 1);
static float64: DType = DType::new(11, 8, "double", Some(3), 1);
static int8: DType = DType::new(1, 1, "char", Some(4), 1);
static int16: DType = DType::new(3, 2, "short", Some(5), 1);
static int32: DType = DType::new(5, 4, "int", Some(6), 1);
static int64: DType = DType::new(7, 8, "long", Some(7), 1);
static uint8: DType = DType::new(2, 1, "unsigned char", Some(8), 1);
static uint16: DType = DType::new(4, 2, "unsigned short", Some(9), 1);
static uint32: DType = DType::new(6, 4, "unsigned int", Some(10), 1);
static uint64: DType = DType::new(8, 8, "unsigned long", Some(11), 1);
static bfloat16: DType = DType::new(9, 2, "__bf16", None, 1);

static DTYPES_DICT: HashMap<i32, DType> = [
    (0, bool),
    (1, float16),
    (2, float32),
    (3, float64),
    (4, int(3, float64)),
    (4, int8),
    (5, int16),
    (6, int32),
    (7, int64),
    (8, uint8),
    (9, uint16),
    (10, uint32),
    (11, uint64),
    (12, bfloat16),
].iter().cloned().collect();

static PtrDtype: Option<DType> = None;
static ImageDType: Option<DType> = None;
static IMAGE: i32 = 0;