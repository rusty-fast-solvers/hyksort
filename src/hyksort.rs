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
    mut kway: i32,
    mut comm: UserCommunicator
) {

    let mut npes = comm.size();
    let mut myrank = comm.rank();

    let mut tot_size: i32 = 0;
    let mut nelem: i32 = arr.len() as i32;

    comm.all_reduce_into(&nelem, &mut tot_size, SystemOperation::sum());

    // Perform local sort
    // arr.sort();

    // Allocate all buffers
    let mut arr_ = vec![0; (nelem*2) as usize];
    // let mut arr__ = vec![0; (nelem*2) as usize];


    while (npes > 1 && tot_size>0) {

        if kway > npes {
            kway = npes;
        }

        // Find size of color block
        let blk_size = npes/kway;
        assert_eq!(blk_size*kway, npes);

        let blk_id = myrank/blk_size;
        let new_pid = modulo(myrank, blk_size);

        // Determine the splitters (placeholder for parallel select algorithm)
        let split_key = vec![1; kway as usize];

        // Communication
        {
            // Determine send size
            let mut send_size: Vec<i32> = vec![0; kway as usize];
            let mut send_disp: Vec<i32> = vec![0; (kway+1) as usize];

            send_disp[0] = 0;
            // Send whole message every time (no bucketing)
            // send_disp[kway as usize] = arr.len() as i32;
            let msg_size = arr.len() as i32;
            send_disp[kway as usize] = msg_size*kway;

            // // Placeholders due to fake split keys
            // for i in 0..kway {
            //     send_disp[i as usize] = (myrank+1)*i;
            // }

            // for i in 0..kway {
            //     send_size[i as usize] = send_disp[(i+1) as usize]-send_disp[i as usize];
            // }

            for i in 0..kway {
                send_size[i as usize] = msg_size;
            }

            for i in 0..kway {
                send_disp[i as usize] = i*msg_size;
            }


            // Testing above code
            // println!("HERE {:?} {:?} {:?}", myrank, send_disp, send_size);

            // Determine receive sizes

            let mut recv_iter: i32 = 0;
            let mut recv_cnt: Vec<i32> = vec![0; kway as usize];
            let mut recv_size: Vec<i32> = vec![0; kway as usize];
            let mut recv_disp: Vec<i32> = vec![0; (kway+1) as usize];

            for i_ in 0..=(kway/2) {
                let i1 = modulo(blk_id+i_, kway);
                let i2 = modulo(blk_id+kway-i_, kway);

                for j in 0..(if i_==0 || i_==kway/2 { 1 }  else {2}) {
                    let i = if i_ == 0  {i1} else { if ((j+blk_id / i_)%2 == 0)  {i1} else {i2} } ;
                    let partner = blk_size*i+new_pid;
                    // println!("rank {:?} partner : {:?} colour {:?}", myrank, partner, blk_id);
                    // println!("rank {:?} i : {:?}", myrank, send_size[i as usize]);
                    let partner_process = comm.process_at_rank(partner);
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
            // println!("RANK {:?} recv_cnt {:?} recv_disp {:?}", myrank, recv_cnt, recv_disp);
            // break;

            // Communicate data
            let asynch_count = 2;
            recv_iter = 0;

            // Resize buffers
            arr_.resize(recv_disp[kway as usize] as usize, 0);
            // arr__.resize(recv_disp[kway as usize] as usize, 0);

            for i_ in 0..=(kway/2) {
                let i1 = modulo(blk_id+i_, kway);
                let i2 = modulo(blk_id+kway-i_, kway);


                for j in 0..(if i_==0 || i_==kway/2 { 1 }  else {2}) {
                    let i = if i_ == 0  {i1} else { if ((j+blk_id / i_)%2 == 0)  {i1} else {i2} } ;
                    let partner = blk_size*i+new_pid;
                    let partner_process = comm.process_at_rank(partner);

                    // println!("rank {:?} receices from {:?} and sends to {:?}", myrank, partner, partner);

                    // Receive indices
                    let r_lidx: usize = recv_disp[recv_iter as usize] as usize;
                    let r_ridx: usize = r_lidx + recv_size[recv_iter as usize] as usize;

                    // Send indices
                    let s_lidx: usize = send_disp[i as usize] as usize;
                    let s_ridx: usize = s_lidx + send_size[i as usize] as usize;

                    mpi::request::scope(|scope| {
                            // let mut sreq = partner_process.immediate_synchronous_send(scope, &arr[s_lidx..s_ridx]);
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
                comm = comm.split_by_color(mpi::topology::Color::with_value(blk_id)).unwrap();
                npes = comm.size();
                myrank = comm.rank();

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