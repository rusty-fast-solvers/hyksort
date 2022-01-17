use std::time::Instant;
use std::collections::HashMap;

use mpi::traits::*;
use mpi::topology::{Color, Rank};
use mpi::collective::{SystemOperation};

use hyksort::hyksort::{hyksort, parallel_select};

pub type Times = HashMap<String, u128>;

use rand::{distributions::Uniform, Rng}; // 0.8.0


fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let mut world = world.split_by_color(Color::with_value(0)).unwrap();
    let size = world.size();
    let rank: Rank = world.rank();
    let k = 128;

    // Sample nparticles randomly in range [min, max)
    let nparticles: u64 = 1000;
    let min = 0;
    let max = 1000000;
    let range = Uniform::from(min..max);
    let mut arr: Vec<u64> = rand::thread_rng()
        .sample_iter(&range)
        .take(nparticles as usize)
        .collect();

    // Store times in a dictionary
    let mut times: Times = HashMap::new();

    let kwayt = Instant::now();
    hyksort(&mut arr, k, &mut world);
    times.insert("kway".to_string(), kwayt.elapsed().as_millis());

    println!(
        "rank {:?} nparticles {:?} min {:?} max {:?}",
        rank, arr.len(), arr.iter().min(), arr.iter().max()
    );

    let mut experiment_size: u64 = 0;
    let world = universe.world();

    // Communicate size of result to master process.
    if rank == 0 {
        world
            .process_at_rank(0)
            .reduce_into_root(&nparticles, &mut experiment_size, SystemOperation::sum());
    } else {
        world
            .process_at_rank(0)
            .reduce_into(&nparticles, SystemOperation::sum());
    }

    if rank == 0 {

        println!("Ranks, Total Particles, Time (ms)");
        println!(
            "{:?}, {:?}, {:?}",
            size,
            experiment_size,
            times.get(&"kway".to_string()).unwrap(),
        );


    }
}
