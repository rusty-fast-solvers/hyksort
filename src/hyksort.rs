use mpi::traits::*;
use mpi::topology::{UserCommunicator, SystemCommunicator};
use mpi::collective::{SystemOperation};
use mpi::datatype::{Partition, PartitionMut};
use mpi::{Count, Address};


// #[cfg(feature = "user-operations")]
// use libffi::middle::{Cif, Closure, Type};

use crate::ffi;
// use crate::ffi::MPI_Op;

pub fn modulo(a: i32, b: i32) -> i32 {
    ((a % b) + b) % b
}

pub fn all_to_all_kwayv(
    arr: &mut Vec<i32>,
    mut k: i32,
    mut comm: UserCommunicator
) {

    let mut p = comm.size();
    let mut rank = comm.rank();

    let mut problem_size: i32 = 0;
    let mut arr_len: i32 = arr.len() as i32;

    comm.all_reduce_into(&arr_len, &mut problem_size, SystemOperation::sum());

    // Allocate all buffers
    let mut arr_ = vec![0; (arr_len*2) as usize];

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
            let mut send_size: Vec<i32> = vec![0; k as usize];
            let mut send_disp: Vec<i32> = vec![0; (k+1) as usize];
            let msg_size = arr.len() as i32;
            // Send whole message every time (no bucketing)
            send_disp[0] = 0;
            send_disp[k as usize] = msg_size*k;

            for i in 0..k {
                send_size[i as usize] = msg_size;
            }

            for i in 0..k {
                send_disp[i as usize] = i*msg_size;
            }

            // Determine receive sizes
            let mut recv_iter: i32 = 0;
            let mut recv_cnt: Vec<i32> = vec![0; k as usize];
            let mut recv_size: Vec<i32> = vec![0; k as usize];
            let mut recv_disp: Vec<i32> = vec![0; (k+1) as usize];

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