use std::convert::TryInto;

use rand::Rng;

use mpi::traits::*;
use mpi::topology::{UserCommunicator, SystemCommunicator};
use mpi::collective::{SystemOperation};
use mpi::datatype::{Partition, PartitionMut};
use mpi::{Count, Address};
use mpi::topology::{Rank, Process};


pub fn modulo(a: i32, b: i32) -> i32 {
    ((a % b) + b) % b
}


pub fn local_rank (iteration: i32, size: i32, k: i32, global_rank: i32) -> i32 {

    modulo(global_rank, (size/ (i32::pow(k, iteration.try_into().unwrap()))))
}


pub fn parallel_select(
    arr: &Vec<u64>,
    &k: &Rank,
    mut comm: &UserCommunicator,
) -> Vec<u64> {
    let mut p: Rank = comm.size();
    let mut rank: Rank = comm.rank();

    let mut problem_size: u64 = 0;
    let mut arr_len: u64 = arr.len().try_into().unwrap();

    comm.all_reduce_into(&arr.len(), &mut problem_size, SystemOperation::sum());

    // Determine number of splitters
    let mut split_count: Count = (1000*k*(arr_len as Count))/(problem_size as Count);
    let mut rng = rand::thread_rng();

    if (p > 1000*k) {
        let num: f64 = rng.gen_range(0.0..1.0);
        if num*(problem_size as f64) < (1000*(k as u64)*problem_size) as f64 {
            split_count = 1;
        } else {
            split_count = 0;
        }
    }

    if split_count > (arr_len as Count) {
        split_count = arr_len as Count;
    }

    // Randomly sample splitters from local section of array
    let mut splitters: Vec<u64> = vec![0; split_count as usize];

    for i in 0..split_count {
        let mut idx: u64 = rng.gen::<u64>();
        idx = idx % arr_len;
        splitters[i as usize] = arr[idx as usize];
    }


    // Gather splitters from all processes
    let mut global_split_count: Count = 0;
    let mut global_split_counts: Vec<Count> = vec![0; p as usize];

    comm.all_gather_into(&split_count, &mut global_split_counts[..]);

    let mut global_split_displacements: Vec<Count> = global_split_counts
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect();

    global_split_count = global_split_displacements[(p-1) as usize] + global_split_counts[(p-1) as usize];

    let mut global_splitters: Vec<u64> = vec![0; global_split_count as usize];
    {
        let mut partition = PartitionMut::new(
            &mut global_splitters[..],
            global_split_counts,
            &global_split_displacements[..]
        );
        comm.all_gather_varcount_into(&splitters[..], &mut partition)
    }

    global_splitters.sort();

    // Rank splitters locally, arr is assumed to be sorted locally
    let mut disp: Vec<u64> = vec![0; (global_split_count as usize)];

    for i in 0..global_split_count {
        disp[i as usize] = arr.partition_point(|&x| x < global_splitters[i as usize]) as u64
    }

    let mut global_disp = vec![0; (global_split_count as usize)];

    for i in 0..global_split_count {
        comm.all_reduce_into(
            &disp[i as usize],
            &mut global_disp[i as usize],
            SystemOperation::sum(),
        );
    }

    let mut split_keys: Vec<u64> = vec![0; k as usize];

    for i in 0..k {
        let mut _disp = 0;
        let optimal_splitter: u64 = ((i+1) as u64 )*problem_size/(k as u64 + 1);

        for j in 0..global_split_count {
            if (global_disp[j as usize]-optimal_splitter as i32).abs() < (global_disp[_disp as usize] - optimal_splitter as i32).abs() {
                _disp = j;
            }
        }

        split_keys[i as usize] = global_splitters[_disp as usize]

    }


    // println!("rank {:?} global_splitters {:?}", rank, disp);
    split_keys
}


pub fn all_to_all_kwayv(
    arr: &mut Vec<u64>,
    mut k: Rank,
    mut comm: UserCommunicator,
) {

    let mut p: Rank = comm.size();
    let mut rank: Rank = comm.rank();

    let mut problem_size: u64 = 0;
    let mut arr_len: u64 = arr.len().try_into().unwrap();

    comm.all_reduce_into(&arr_len, &mut problem_size, SystemOperation::sum());

    // Allocate all buffers
    let mut arr_: Vec<u64> = vec![0; (arr_len*2) as usize];

    let mut iter = 0;

    let global_rank = rank;
    let global_size = comm.size();

    // Local sort
    arr.sort();

    while (p > 1 && problem_size>0) {

        if k > p {
            k = p;
        }

        // Find size of color block
        let color_size = p/k;
        assert_eq!(color_size*k, p);

        let color = rank/color_size;
        let new_rank = modulo(rank, color_size);

        // Find splitters
        let split_keys: Vec<u64> = parallel_select(&arr, &(k-1), &comm);

        // Communicate
        {
            // Determine send size
            let mut send_size: Vec<u64> = vec![0; k as usize];
            let mut send_disp: Vec<u64> = vec![0; (k+1) as usize];
            send_disp[k as usize] = arr.len() as u64;

            for i in 1..k {
                send_disp[i as usize] = arr.partition_point(|&x| x < split_keys[(i-1) as usize]) as u64;
            }

            for i in 0..k {
                send_size[i as usize] = send_disp[(i+1) as usize] - send_disp[i as usize];
            }

            // if rank == 0 {
            //     println!("split keys {:?}", split_keys);
            //     println!("send disp {:?}", send_disp);
            //     println!("send size {:?}", send_size);
            // }

            // break;
            // let msg_size: u64 = arr.len() as u64;

            // // Send whole message every time (no bucketing)
            // send_disp[0] = 0;
            // send_disp[k as usize] = msg_size*(k as u64);

            // for i in 0..k {
            //     send_size[i as usize] = msg_size;
            // }

            // for i in 0..k {
            //     send_disp[i as usize] = (i as u64)*msg_size;
            // }

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
                            let mut sreq = partner_process.immediate_synchronous_send(scope, &arr[s_lidx..s_ridx]);
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

        iter += 1;
    }
}

