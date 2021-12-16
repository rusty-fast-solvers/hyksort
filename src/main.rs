use std::time::Instant;
use std::collections::HashMap;

use mpi::traits::*;
use mpi::topology::{Color};

use hyksort::hyksort::{all_to_all_kwayv, all_to_all};

pub type Times = HashMap<String, u128>;


fn main() {

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let world = world.split_by_color(Color::with_value(0)).unwrap();
    let size = world.size();
    let rank = world.rank();
    let k = 2;

    let nparticles = 1e5;
    let mut arr = vec![rank; nparticles as usize];

    let mut buckets: Vec<Vec<i32>> = vec![Vec::new(); (size-1) as usize];

    for i in 0..(size-1) {
        for elem in &arr {
            buckets[i as usize].push(elem.clone());
        }
    }

    let mut times: Times = HashMap::new();

    let kwayt = Instant::now();
    all_to_all_kwayv(&mut arr, k, world);
    times.insert("kway".to_string(), kwayt.elapsed().as_millis());

    let world = universe.world();
    let intrinsic = Instant::now();
    let mut b = all_to_all(world, size, buckets);
    times.insert("intrinsic".to_string(), intrinsic.elapsed().as_millis());

    if rank == 0 {
        println!(
            "{:?} {:?}, {:?}, {:?}",
            size,
            nparticles,
            times.get(&"kway".to_string()).unwrap(),
            times.get(&"intrinsic".to_string()).unwrap()
        );
        assert_eq!(arr.len(), b.len());
        arr.sort();
        b.sort();
        assert_eq!(arr, b);

        // println!("a {:?} b {:?}", arr, b);
    }
}
