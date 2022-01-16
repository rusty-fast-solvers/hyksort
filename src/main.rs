use std::time::Instant;
use std::collections::HashMap;

use mpi::traits::*;
use mpi::topology::{Color, Rank};

use hyksort::hyksort::{all_to_all_kwayv, parallel_select};

pub type Times = HashMap<String, u128>;


fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let world = world.split_by_color(Color::with_value(0)).unwrap();
    let size = world.size();
    let rank: Rank = world.rank();
    let k = 128;
    let k = 2;

    let nparticles = 10;
    let mut arr = vec![rank as u64; nparticles as usize];

    let mut arr: Vec<u64> = vec![0, 3, 1, 2, 0, 3, 1, 2];
    arr.sort();
    // println!("arr {:?}", arr);
    let mut buckets: Vec<Vec<u64>> = vec![Vec::new(); (size-1) as usize];

    for i in 0..(size-1) {
        for elem in &arr {
            buckets[i as usize].push(elem.clone());
        }
    }

    let mut times: Times = HashMap::new();
    let world = universe.world();
    let world = world.split_by_color(Color::with_value(0)).unwrap();

    // let split_keys = parallel_select(&mut arr, &(k-1), &world);

    // println!("rank {:?} split keys {:?}", rank, split_keys);


    // Hyksort communication pattern
    let kwayt = Instant::now();
    let res = all_to_all_kwayv(&mut arr, k, world);
    times.insert("kway".to_string(), kwayt.elapsed().as_millis());

    println!("rank {:?} res {:?}", rank, arr);
    // let world = universe.world();
    // let intrinsic = Instant::now();
    // // let mut b = all_to_all(world, size, buckets);
    // times.insert("intrinsic".to_string(), intrinsic.elapsed().as_millis());


    // if rank == 0 {
    //     println!(
    //         "{:?} {:?}, {:?}, {:?}",
    //         size,
    //         nparticles,
    //         times.get(&"kway".to_string()).unwrap(),
    //         times.get(&"intrinsic".to_string()).unwrap()
    //     );
    //     // assert_eq!(arr.len(), b.len());
    // }
}
