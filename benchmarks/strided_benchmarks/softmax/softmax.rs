//! Fused axis-0 softmax throughput for `strided-nn`.
//!
//! The kernel takes contiguous column-major slices, caller scratch and an
//! explicit `Exec` placement token, so this bench reports the time of one call
//! at a fixed thread budget, not of an engine pipeline around it.
//!
//! Run thread-count variants one at a time; never run benchmark processes in
//! parallel. Set `THREADS` (default 1), `AXIS_LENS`, `COLUMNS` and `RUNS` to
//! narrow the matrix.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use strided_kernel::{Exec, Pool};
use strided_nn::{scratch_len, softmax_axis0_f32, softmax_axis0_f64};

/// Counts allocations so steady-state calls can be shown to allocate nothing.
struct CountingAlloc;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAlloc = CountingAlloc;

static POOL: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();

/// Mint a fresh placement token (tokens are not `Clone`); the pool is built once.
fn exec(threads: usize) -> Exec<'static> {
    if threads == 1 {
        return Exec::serial();
    }
    let pool = POOL.get_or_init(|| {
        Pool::builder()
            .threads(threads)
            .build()
            .expect("bench pool")
    });
    Exec::new(pool, threads).expect("exec")
}

#[derive(Clone, Copy)]
struct Stats {
    median_ms: f64,
    iqr_ms: f64,
    allocations: usize,
}

fn parse_list(name: &str, default: &[usize]) -> Vec<usize> {
    let Ok(value) = std::env::var(name) else {
        return default.to_vec();
    };
    if value.trim() == "all" {
        return default.to_vec();
    }
    value
        .split(',')
        .map(|entry| entry.trim().parse().expect("invalid list entry"))
        .collect()
}

fn stats(samples: &mut [Duration]) -> (f64, f64) {
    samples.sort();
    let median = samples[samples.len() / 2].as_secs_f64() * 1e3;
    let p25 = samples[samples.len() / 4].as_secs_f64() * 1e3;
    let p75 = samples[samples.len() * 3 / 4].as_secs_f64() * 1e3;
    (median, p75 - p25)
}

fn bench(mut call: impl FnMut(), warmup: usize, runs: usize) -> Stats {
    for _ in 0..warmup {
        call();
    }
    let mut samples = Vec::with_capacity(runs);
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..runs {
        let start = Instant::now();
        call();
        samples.push(start.elapsed());
    }
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - before;
    let (median_ms, iqr_ms) = stats(&mut samples);
    Stats {
        median_ms,
        iqr_ms,
        allocations,
    }
}

fn main() {
    let threads = std::env::var("THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1usize);
    let axis_lens = parse_list("AXIS_LENS", &[64, 256, 1024, 4096, 16384]);
    let columns_list = parse_list("COLUMNS", &[1, 64, 1024]);
    let runs = std::env::var("RUNS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20usize);
    // Above this many elements one case would dominate the run time.
    let max_elements: usize = std::env::var("MAX_ELEMENTS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1 << 24);

    // Build the pool and mint one token before any timed region, so the first
    // sample does not pay for the pool's one-time allocations.
    let _ = black_box(exec(threads));

    println!(
        "| dtype | axis_len | columns | elements | threads | median_ms | iqr_ms | GB/s | allocations |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    for &axis_len in &axis_lens {
        for &columns in &columns_list {
            let elements = axis_len * columns;
            if elements > max_elements {
                continue;
            }
            let row_f32 = measure_f32(threads, axis_len, columns, runs);
            report(
                "f32",
                axis_len,
                columns,
                elements,
                threads,
                row_f32,
                elements * 8,
            );
            let row_f64 = measure_f64(threads, axis_len, columns, runs);
            report(
                "f64",
                axis_len,
                columns,
                elements,
                threads,
                row_f64,
                elements * 16,
            );
        }
    }
}

fn report(
    dtype: &str,
    axis_len: usize,
    columns: usize,
    elements: usize,
    threads: usize,
    stats: Stats,
    bytes: usize,
) {
    let gb_per_s = if stats.median_ms > 0.0 {
        bytes as f64 / (stats.median_ms * 1e-3) / 1e9
    } else {
        0.0
    };
    println!(
        "| {dtype} | {axis_len} | {columns} | {elements} | {threads} | {:.3} | {:.3} | {:.2} | {} |",
        stats.median_ms, stats.iqr_ms, gb_per_s, stats.allocations
    );
}

fn measure_f32(threads: usize, axis_len: usize, columns: usize, runs: usize) -> Stats {
    let len = axis_len * columns;
    let input: Vec<f32> = (0..len)
        .map(|i| ((i % axis_len) as f32) * 0.01 - 0.5)
        .collect();
    let mut output = vec![std::mem::MaybeUninit::<f32>::uninit(); len];
    let mut scratch = vec![0.0f32; scratch_len::<f32>(axis_len, len, threads)];
    bench(
        || {
            softmax_axis0_f32(
                axis_len,
                black_box(&input),
                &mut output,
                &mut scratch,
                exec(threads),
            )
            .expect("valid softmax");
        },
        3,
        runs,
    )
}

fn measure_f64(threads: usize, axis_len: usize, columns: usize, runs: usize) -> Stats {
    let len = axis_len * columns;
    let input: Vec<f64> = (0..len)
        .map(|i| ((i % axis_len) as f64) * 0.01 - 0.5)
        .collect();
    let mut output = vec![std::mem::MaybeUninit::<f64>::uninit(); len];
    let mut scratch = vec![0.0f64; scratch_len::<f64>(axis_len, len, threads)];
    bench(
        || {
            softmax_axis0_f64(
                axis_len,
                black_box(&input),
                &mut output,
                &mut scratch,
                exec(threads),
            )
            .expect("valid softmax");
        },
        3,
        runs,
    )
}
