extern crate hyksort;
extern crate mpi;

use mpi::environment::Universe;
use mpi::topology::{Color, Rank};
use mpi::traits::*;

use rand::{distributions::Uniform, Rng};

use hyksort::hyksort::{hyksort, parallel_select};

pub fn test_hyksort(universe: &Universe) {
    let world = universe.world();
    let size = world.size();
    let rank: Rank = world.rank();
    let k = 4;

    // Sample nparticles randomly in range [min, max)
    let nparticles: u64 = 1000;
    let min = 0;
    let max = 10000000000;
    let range = Uniform::from(min..max);
    let mut arr: Vec<u64> = rand::thread_rng()
        .sample_iter(&range)
        .take(nparticles as usize)
        .collect();

    hyksort(&mut arr, k, &world);

    // Test that the minimum on this process is greater than the maximum on the previous process
    let prev_rank = if rank > 0 { rank - 1 } else { size - 1 };
    let prev_process = world.process_at_rank(prev_rank);

    if rank > 0 {
        let min: u64 = arr.iter().min().unwrap().clone();
        prev_process.send(&min);
    }

    if rank < (size - 1) {
        let (rec, _) = world.any_process().receive_vec::<u64>();
        let max: u64 = arr.iter().max().unwrap().clone();
        assert!(max <= rec[0], "{:?} {:?}", max, rec);
    }

    // Test that array is sorted on each process
    let mut prev = arr[0];
    for &elem in arr.iter().skip(1) {
        assert!(elem >= prev);
        prev = elem;
    }

    if rank == 0 {
        println!("... HykSort Passed!")
    }
}

pub fn test_parallel_select(universe: &Universe) {
    let world = universe.world();
    let rank: Rank = world.rank();

    if rank == 0 {
        println!("... Parallel Select Passed!")
    }
}
