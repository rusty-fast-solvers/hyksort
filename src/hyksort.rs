use mpi::traits::*;
use mpi::topology::{UserCommunicator, SystemCommunicator};
use mpi::collective::{SystemOperation};

// #[cfg(feature = "user-operations")]
// use libffi::middle::{Cif, Closure, Type};

use crate::ffi;
// use crate::ffi::MPI_Op;

pub fn modulo(a: i32, b: i32) -> i32 {
    ((a % b) + b) % b
}

pub fn hyksort(
    arr: &mut [i32],
    mut kway: i32,
    comm: UserCommunicator
) {

    let npes = comm.size();
    let myrank = comm.rank();

    let mut tot_size: i32 = 0;
    let mut nelem: i32 = arr.len() as i32;

    comm.all_reduce_into(&nelem, &mut tot_size, SystemOperation::sum());

    // Perform local sort
    arr.sort();

    // Allocate all buffers
    let mut arr_ = vec![0; (nelem*2) as usize];
    let mut arr__ = vec![0; (nelem*2) as usize];


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
            send_disp[kway as usize] = arr.len() as i32;

            // Placeholders due to fake split keys
            for i in 0..kway {
                send_disp[i as usize] = (myrank+1)*i;
            }

            for i in 0..kway {
                send_size[i as usize] = send_disp[(i+1) as usize]-send_disp[i as usize];
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
                    // println!("rank {:?} partner : {:?}", myrank, partner);
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
            println!("RANK {:?} recv_cnt {:?} recv_disp {:?}", myrank, recv_cnt, recv_disp);

            // Communicate data
            break;



        }

    }
}