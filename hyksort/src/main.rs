use std::collections::HashMap;
/// Simple script to test weak scaling of Hyksort algorithm
use std::time::Instant;

use mpi::collective::SystemOperation;
use mpi::topology::{Color, Rank};
use mpi::traits::*;

use hyksort::hyksort::{hyksort, parallel_select};

pub type Times = HashMap<String, u128>;

use rand::{distributions::Uniform, Rng};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let mut world = world.split_by_color(Color::with_value(0)).unwrap();
    let size = world.size();
    let rank: Rank = world.rank();
    let k = 2;

    // Sample nparticles randomly in range [min, max)
    let nparticles: u64 = 10000000;
    // let nparticles: u64 = 1000;
    let min = 0;
    let max = 10000000000;
    let range = Uniform::from(min..max);
    let mut arr: Vec<u64> = rand::thread_rng()
        .sample_iter(&range)
        .take(nparticles as usize)
        .collect();

//     // Store times in a dictionary
    // let mut times: Times = HashMap::new();

    let kwayt = Instant::now();
    let profile = hyksort(&mut arr, k, &mut world);
    let wall_time = kwayt.elapsed().as_millis() as u64;
    // times.insert("kway".to_string(), kwayt.elapsed().as_millis());

    // println!("Rank {:?} \n Profile: {:?} \n Wall Time: {:?} \n", rank, profile, wall_time);

    let world = universe.world();
    let mut world = world.split_by_color(Color::with_value(0)).unwrap();

    // Times stored as milliseconds
    if rank == 0 {
        let mut total_wall_time: u64 = 0;
        let mut total: u64 = 0;
        let mut local_sort: u64 = 0;
        let mut communication: u64 = 0;

        world
            .process_at_rank(0)
            .reduce_into_root(&wall_time, &mut total_wall_time, SystemOperation::sum());

        world
            .process_at_rank(0)
            .reduce_into_root(&profile.total, &mut total, SystemOperation::sum());

        world
            .process_at_rank(0)
            .reduce_into_root(&profile.local_sort, &mut local_sort, SystemOperation::sum());

        world
            .process_at_rank(0)
            .reduce_into_root(&profile.communication, &mut communication, SystemOperation::sum());

            println!("K={:?}, nparticles={:?}", k, nparticles*(size as u64));
            println!(" Mean Wall Time {:?} ms", (total_wall_time as f64)/(size as f64));
            println!(" Mean CPU Time {:?} ms", (total as f64)/(size as f64));
            println!(" Mean Local Sort Time {:?} ms", (local_sort as f64)/(size as f64));
            println!(" Mean Communcation Time {:?} ms", (communication as f64)/(size as f64));
    } else {

        world
            .process_at_rank(0)
            .reduce_into(&wall_time, SystemOperation::sum());

            world
            .process_at_rank(0)
            .reduce_into(&profile.total, SystemOperation::sum());

        world
            .process_at_rank(0)
            .reduce_into(&profile.local_sort, SystemOperation::sum());

        world
            .process_at_rank(0)
            .reduce_into(&profile.communication, SystemOperation::sum());
    }

}
