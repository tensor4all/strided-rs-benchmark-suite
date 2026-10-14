//! Paired benchmark for strided-rs issue #300: the raw copy/scale/axpy rank cliff.
//!
//! Prints `op,rank,kind,total,fused_rank,ns_per_elem,allocs_per_call,bytes_per_call`
//! per case. One process = one repetition; the driver alternates baseline and
//! candidate processes and aggregates per-case minima.
//!
//! Usage: rawrank300 [TOTAL...]   (default: 4096 1048576)

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(l.size(), Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(n, Ordering::Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(l.size(), Ordering::Relaxed);
        unsafe { System.alloc_zeroed(l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

use strided_basic::{axpy_raw, copy_scale_raw, CopyPlan, RawStridedMut, RawStridedRef};

fn dims_for(rank: usize, total: usize) -> Vec<usize> {
    let mut d = vec![1usize; rank];
    let mut rem = total;
    for i in 0..rank {
        d[i] = 2.min(rem);
        rem /= d[i];
    }
    d[rank - 1] *= rem.max(1);
    d
}

fn contig_strides(dims: &[usize]) -> Vec<isize> {
    let mut s = Vec::with_capacity(dims.len());
    let mut next = 1isize;
    for &d in dims {
        s.push(next);
        next *= d as isize;
    }
    s
}

/// Column-major strides with exactly one padding element between consecutive blocks.
fn padded_strides(dims: &[usize]) -> Vec<isize> {
    let mut s = Vec::with_capacity(dims.len());
    let mut next = 1isize;
    for &d in dims {
        s.push(next);
        next = next * d as isize + 1;
    }
    s
}

/// Reference model of the library's fusion rule: drop size-1 axes, sort by
/// destination stride, merge axes contiguous in both layouts.
fn fused_rank(dims: &[usize], dst: &[isize], src: &[isize]) -> usize {
    let mut axes: Vec<(usize, isize, isize)> = dims
        .iter()
        .enumerate()
        .filter(|(_, &d)| d != 1)
        .map(|(i, &d)| (d, dst[i], src[i]))
        .collect();
    if axes.is_empty() {
        return 1;
    }
    axes.sort_by_key(|a| a.1);
    let mut fused = 0usize;
    for i in 1..axes.len() {
        let (pd, pdst, psrc) = axes[fused];
        if (pd as isize).checked_mul(pdst) == Some(axes[i].1)
            && (pd as isize).checked_mul(psrc) == Some(axes[i].2)
        {
            axes[fused].0 = pd * axes[i].0;
        } else {
            fused += 1;
            axes[fused] = axes[i];
        }
    }
    fused + 1
}

fn span(dims: &[usize], strides: &[isize]) -> usize {
    let mut max = 0isize;
    let mut min = 0isize;
    for (&d, &s) in dims.iter().zip(strides) {
        if d == 0 {
            continue;
        }
        let e = (d as isize - 1) * s;
        if e > 0 { max += e } else { min += e }
    }
    (max - min + 1) as usize
}

struct Case {
    rank: usize,
    kind: &'static str,
    dims: Vec<usize>,
    dst: Vec<isize>,
    src: Vec<isize>,
    dst_offset: isize,
    src_offset: isize,
    dst_len: usize,
    src_len: usize,
}

fn build_cases(total: usize, ranks: &[usize]) -> Vec<Case> {
    let mut out = Vec::new();
    for &rank in ranks {
        let base = dims_for(rank, total);
        // contiguous (fuses to one axis)
        let s = contig_strides(&base);
        out.push(Case { rank, kind: "contig", dims: base.clone(), dst: s.clone(), src: s.clone(), dst_offset: 0, src_offset: 0, dst_len: span(&base, &s), src_len: span(&base, &s) });
        // padded, inner axis short (2)
        let ps = padded_strides(&base);
        out.push(Case { rank, kind: "padded-short-inner", dims: base.clone(), dst: ps.clone(), src: ps.clone(), dst_offset: 0, src_offset: 0, dst_len: span(&base, &ps), src_len: span(&base, &ps) });
        // padded, inner axis long (reverse the axis order)
        let mut revd = base.clone();
        revd.reverse();
        let prs = padded_strides(&revd);
        out.push(Case { rank, kind: "padded-long-inner", dims: revd.clone(), dst: prs.clone(), src: prs.clone(), dst_offset: 0, src_offset: 0, dst_len: span(&revd, &prs), src_len: span(&revd, &prs) });
        // reversed source over a contiguous destination
        let off: isize = base.iter().zip(&s).map(|(&d, &st)| (d as isize - 1) * st).sum();
        out.push(Case { rank, kind: "reversed-src", dims: base.clone(), dst: s.clone(), src: s.iter().map(|&x| -x).collect(), dst_offset: 0, src_offset: off, dst_len: span(&base, &s), src_len: span(&base, &s) });
    }
    out
}

fn time_ns<F: FnMut()>(iters: usize, mut f: F) -> (f64, usize, usize) {
    for _ in 0..iters.min(8) {
        f();
    }
    let a0 = ALLOCS.load(Ordering::Relaxed);
    let b0 = BYTES.load(Ordering::Relaxed);
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    let dt = t0.elapsed().as_nanos() as f64 / iters as f64;
    (
        dt,
        (ALLOCS.load(Ordering::Relaxed) - a0) / iters,
        (BYTES.load(Ordering::Relaxed) - b0) / iters,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let totals: Vec<usize> = if args.is_empty() {
        vec![4096, 1 << 20]
    } else {
        args.iter().map(|s| s.parse().unwrap()).collect()
    };
    let ranks = [2usize, 4, 6, 8, 9, 12, 16];
    println!("op,rank,kind,total,fused_rank,ns_per_elem,allocs_per_call,bytes_per_call");
    for &total in &totals {
        for case in build_cases(total, &ranks) {
            let n = case.dims.iter().product::<usize>();
            let fr = fused_rank(&case.dims, &case.dst, &case.src);
            let src = vec![1.0f64; case.src_len];
            let mut dst = vec![1.0f64; case.dst_len];
            let iters = (30_000_000 / n.max(1)).max(4);

            let r = time_ns(iters, || {
                let mut d = RawStridedMut::new(&mut dst, &case.dims, &case.dst, case.dst_offset).unwrap();
                let s = RawStridedRef::new(&src, &case.dims, &case.src, case.src_offset).unwrap();
                axpy_raw(&mut d, &s, 1.0).unwrap();
            });
            println!("axpy_raw,{},{},{},{},{:.4},{},{}", case.rank, case.kind, n, fr, r.0 / n as f64, r.1, r.2);

            let r = time_ns(iters, || {
                let mut d = RawStridedMut::new(&mut dst, &case.dims, &case.dst, case.dst_offset).unwrap();
                let s = RawStridedRef::new(&src, &case.dims, &case.src, case.src_offset).unwrap();
                copy_scale_raw(&mut d, &s, 1.0).unwrap();
            });
            println!("copy_scale_raw,{},{},{},{},{:.4},{},{}", case.rank, case.kind, n, fr, r.0 / n as f64, r.1, r.2);

            let plan = CopyPlan::compile(&case.dims, &case.dst, &case.src).unwrap();
            let r = time_ns(iters, || {
                let mut d = RawStridedMut::new(&mut dst, &case.dims, &case.dst, case.dst_offset).unwrap();
                let s = RawStridedRef::new(&src, &case.dims, &case.src, case.src_offset).unwrap();
                plan.execute(&mut d, &s).unwrap();
            });
            println!("CopyPlan::execute,{},{},{},{},{:.4},{},{}", case.rank, case.kind, n, fr, r.0 / n as f64, r.1, r.2);
        }
    }
}
