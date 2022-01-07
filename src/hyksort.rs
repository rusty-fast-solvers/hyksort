use std::convert::TryInto;

use rand::{thread_rng, Rng};

use mpi::traits::*;
use mpi::topology::{UserCommunicator, SystemCommunicator};
use mpi::collective::{SystemOperation};
use mpi::datatype::{Partition, PartitionMut};
use mpi::{Count, Address};
use mpi::topology::{Rank, Process};


pub fn modulo(a: i32, b: i32) -> i32 {
    ((a % b) + b) % b
}

pub fn select_splitters(
    arr: &mut Vec<u64>,
    mut k: Rank,
    mut comm: UserCommunicator
 ) -> Vec<u64> {
    let mut rank: Rank = comm.rank();
    let mut size: Rank = comm.size();

    if k > size {
        k = size;
    }

    let arr_len: u64 = arr.len().try_into().unwrap();
    let mut problem_size: u64 = 0;

    // Find total problem size
    comm.all_reduce_into(&arr_len, &mut problem_size, SystemOperation::sum());

    // 1. Collect samples from each process onto all other processes
    let n_samples: usize = 10;

    let mut rng = thread_rng();
    let sample_idxs: Vec<usize> = (0..n_samples)
        .map(|_| rng.gen_range(0..arr_len as usize))
        .collect();

    let mut local_samples: Vec<u64> = vec![0 as u64; n_samples];
    let mut received_samples: Vec<u64> = vec![0; n_samples * (size as usize)];

    for (i, &sample_idx) in sample_idxs.iter().enumerate() {
        local_samples[i] = arr[sample_idx].clone();
    }

    comm.all_gather_into(&local_samples[..], &mut received_samples[..]);

    // We want 'k' splitters to define k+1 buckets
    // let total_samples: u64 = n_samples*size;
    let n_buckets: usize = ((n_samples) * (size as usize)) / (k as usize);

    received_samples.sort();
    // println!("ALL RECEIVED {:?}", n_buckets);

    let splitters = received_samples.iter().step_by(n_buckets).cloned().collect();

    splitters
}


pub fn all_to_all_kwayv(
    arr: &mut Vec<u64>,
    mut k: Rank,
    splitters: &Vec<u64>,
    mut comm: UserCommunicator,
) {

    let mut p: Rank = comm.size();
    let mut rank: Rank = comm.rank();

    let mut problem_size: u64 = 0;
    let mut arr_len: u64 = arr.len().try_into().unwrap();

    comm.all_reduce_into(&arr_len, &mut problem_size, SystemOperation::sum());

    // Allocate all buffers
    let mut arr_: Vec<u64> = vec![0; (arr_len*2) as usize];

    while (p > 1 && problem_size>0) {

        if k > p {
            k = p;
        }

        // Find size of color block
        let color_size = p/k;
        assert_eq!(color_size*k, p);

        let color = rank/color_size;
        let new_rank = modulo(rank, color_size);

        // Communicate
        {
            // Determine send size
            let mut send_size: Vec<u64> = vec![0; k as usize];
            let mut send_disp: Vec<u64> = vec![0; (k+1) as usize];
            let msg_size: u64 = arr.len() as u64;

            // Send whole message every time (no bucketing)
            send_disp[0] = 0;
            send_disp[k as usize] = msg_size*(k as u64);

            for i in 0..k {
                send_size[i as usize] = msg_size;
            }

            for i in 0..k {
                send_disp[i as usize] = (i as u64)*msg_size;
            }

            // Determine receive sizes
            let mut recv_iter: u64 = 0;
            let mut recv_cnt: Vec<u64> = vec![0; k as usize];
            let mut recv_size: Vec<u64> = vec![0; k as usize];
            let mut recv_disp: Vec<u64> = vec![0; (k+1) as usize];

            for i_ in 0..=(k/2) {
                let i1 = modulo(color+i_, k);
                let i2 = modulo(color+k-i_, k);

                for j in 0..(if i_==0 || i_==k/2 { 1 }  else {2}) {
                    let i = if i_ == 0  {i1} else { if ((j+color / i_)%2 == 0)  {i1} else {i2} } ;
                    let partner_rank = color_size*i+new_rank;
                    let partner_process = comm.process_at_rank(partner_rank);
                    mpi::point_to_point::send_receive_into(
                        &send_size[i as usize],
                        &partner_process,
                        &mut recv_size[recv_iter as usize],
                        &partner_process
                    );
                    recv_disp[(recv_iter+1) as usize] = recv_disp[recv_iter as usize]+recv_size[recv_iter as usize];
                    recv_cnt[recv_iter as usize] = recv_size[recv_iter as usize];
                    recv_iter += 1;
                }
            }

            // Communicate data
            recv_iter = 0;

            // Resize buffers
            arr_.resize(recv_disp[k as usize] as usize, 0);

            for i_ in 0..=(k/2) {
                let i1 = modulo(color+i_, k);
                let i2 = modulo(color+k-i_, k);

                for j in 0..(if i_==0 || i_==k/2 { 1 }  else {2}) {
                    let i = if i_ == 0  {i1} else { if ((j+color / i_)%2 == 0)  {i1} else {i2} } ;
                    let partner_rank = color_size*i+new_rank;
                    let partner_process = comm.process_at_rank(partner_rank);

                    // Receive indices
                    let r_lidx: usize = recv_disp[recv_iter as usize] as usize;
                    let r_ridx: usize = r_lidx + recv_size[recv_iter as usize] as usize;

                    // Send indices
                    let s_lidx: usize = send_disp[i as usize] as usize;
                    let s_ridx: usize = s_lidx + send_size[i as usize] as usize;

                    mpi::request::scope(|scope| {
                            let mut sreq = partner_process.immediate_synchronous_send(scope, &arr[..]);
                            let rreq = partner_process.immediate_receive_into(scope, &mut arr_[r_lidx..r_ridx]);
                            rreq.wait();
                            loop {
                                match sreq.test() {
                                    Ok(_) => {
                                        break;
                                    } Err(req) => {
                                        sreq = req;
                                    }
                                }
                            }
                    });
                    recv_iter += 1;
                }
            }
            // Swap buffers
            std::mem::swap(arr, &mut arr_);

            // Handle communicator
            {
                comm = comm.split_by_color(mpi::topology::Color::with_value(color)).unwrap();
                p = comm.size();
                rank = comm.rank();

            }
        }
    }
}



pub fn all_to_all<T>(
    world: SystemCommunicator,
    size: mpi::topology::Rank,
    buckets: Vec<Vec<T>>) -> Vec<T>
where T: Default+Clone+Equivalence
{

    let mut counts_snd: Vec<Count> = vec![0; size as usize];

    for (i, bucket) in buckets.iter().enumerate() {
        counts_snd[i] = bucket.len() as Count;
    }

    // Flatten buckets, after bucketing
    let buckets_flat: Vec<T> = buckets.into_iter().flatten().collect();

    let displs_snd: Vec<Count> = counts_snd
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect();

    // All to All for bucket sizes
    let mut counts_recv: Vec<Count> = vec![0; size as usize];

    world.all_to_all_into(&counts_snd[..], &mut counts_recv[..]);

    // displacements
    let displs_recv: Vec<Count> = counts_recv
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect();

    // Allocate a buffer to receive relevant data from all processes.
    let total: Count = counts_recv.iter().sum();
    let mut received = vec![T::default(); total as usize];
    let mut partition_receive = PartitionMut::new(&mut received[..], counts_recv, &displs_recv[..]);

    // Allocate a partition of the data to send to each process
    let partition_snd = Partition::new(&buckets_flat[..], counts_snd, &displs_snd[..]);

    world.all_to_all_varcount_into(&partition_snd, &mut partition_receive);

    received
}