use mpi::traits::*;
use hyksort::hyksort;
use mpi::topology::{Color};

fn main() {

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let world = world.split_by_color(Color::with_value(0)).unwrap();
    let rank = world.rank();
    let kway = 4;

    let mut arr = vec![rank; (kway*(rank+1)) as usize];
    hyksort::hyksort(&mut arr, kway, world);
}
