/// Test suite and runner for MPI based programs
use mpi::traits::*;
use parallel_tests::sorting::{test_hyksort, test_parallel_select};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    if rank == 0 {
        println!("Testing HykSort: ");
    };
    test_hyksort(&universe);

    let world = universe.world();
    let rank = world.rank();

    if rank == 0 {
        println!("Testing Parallel Select: ");
    };

    test_parallel_select(&universe);
}
