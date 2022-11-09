/// Tests for Parallel Select and Hyksort algorithms.
extern crate hyksort;
extern crate mpi;

use mpi::environment::Universe;
use mpi::collective::SystemOperation;
use mpi::topology::{Rank, SystemCommunicator};
use mpi::traits::*;

use rand::{distributions::Uniform, Rng};

use hyksort::hyksort::{hyksort, parallel_select};

pub fn test_hyksort(world: &SystemCommunicator) {
    let comm = world.duplicate();
    let size = world.size();
    let rank: Rank = world.rank();
    let k = 2;

    // Sample nparticles randomly in range [min, max)
    let nparticles: u64 = 100000;
    let min = 0;
    let max = 10000000000;
    let range = Uniform::from(min..max);
    let mut arr: Vec<u64> = rand::thread_rng()
        .sample_iter(&range)
        .take(nparticles as usize)
        .collect();

    hyksort(&mut arr, k, comm);

    // Test that the minimum on this process is greater than the maximum on the previous process
    let prev_rank = if rank > 0 { rank - 1 } else { size - 1 };
    let prev_process = world.process_at_rank(prev_rank);

    if rank > 0 {
        let min: u64 = *arr.iter().min().unwrap();
        prev_process.send(&min);
    }

    if rank < (size - 1) {
        let (rec, _) = world.any_process().receive_vec::<u64>();
        let max: u64 = *arr.iter().max().unwrap();
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

    // Test that no elements of the distributed array are lost
    let mut problem_size = 0;
    world.all_reduce_into(&arr.len(), &mut problem_size, SystemOperation::sum());

    assert!(problem_size == (size as u64)*nparticles);

}

pub fn test_parallel_select(world: &SystemCommunicator) {
    let comm = world.duplicate();
    let size = world.size();
    let rank: Rank = world.rank();
    let k = 4;

    // Sample nparticles randomly in range [min, max)
    let nparticles: u64 = 10000;
    let min = 0;
    let max = 10000000000;
    let range = Uniform::from(min..max);
    let mut arr: Vec<u64> = rand::thread_rng()
        .sample_iter(&range)
        .take(nparticles as usize)
        .collect();

    let splitters = parallel_select(&arr, &k, comm);

    // Test that the number of splitters matches 'K'
    assert!(splitters.len() == k as usize);

    // Test that the splitters are within the range of the array.
    for splitter in splitters {
        assert!(min <= splitter && splitter <= max);
    }

    if rank == 0 {
        println!("... Parallel Select Passed!")
    }
}
