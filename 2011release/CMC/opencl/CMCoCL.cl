/* OpenCL kernels for Classic Monte Carlo on the GPU */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#include "ranluxcl.cl"

/*
 Call once to ensure decorrelation of the random number generator
 */
__kernel void Kernel_RANLUXCL_Warmup(__global float4* RANLUXCLTab) {
    //Downloading RANLUXCLTab. The state of RANLUXCL is stored in ranluxclstate.
    ranluxcl_state_t ranluxclstate;
    ranluxcl_download_seed(&ranluxclstate, RANLUXCLTab);

    ranluxcl_warmup(&ranluxclstate);

    //Uploading RANLUXCLTab
    ranluxcl_upload_seed(&ranluxclstate, RANLUXCLTab);
}

struct sysdata {
  float NP;
  int nx;
  float lx;
  float dx;
  float dx_recip;
  float sig_t;
  float sig_t_recip;
  float sig_s;
  float eps;
  float q0_avg;
  float NPc;
};

//Kernel to stream the particles and collect statistics
//Number of blocks = number of cells in the domain
//Number of threads = number of cells in the domain
//Each workgroup craetes particles in one cell of the domain and streams it, keeping track of its contributions to all cells in the domain
//The tallying is done as follows : 
//After streaming a particle till it collides once/gets absorbed /goes out
// Step 0 : thread 0 adds contribution to cell 0, thread 1 adds contribution to cell 1.... thread ncells -1 adds contribution to cell ncells-1
// Step 1 : thread 0 adds contribution to cell 1, thread 1 adds contribution to cell 2.... thread ncells -1 adds contribution to cell 0
// Step 2 : thread 0 adds contribution to cell 2, thread 1 adds contribution to cell 3.... thread ncells -1 adds contribution to cell 1
// .....
// Step ncells - 1 : thread 0 adds contribution to cell ncells - 1 , thread 1 adds contribution to cell 0.... thread ncells -1 adds contribution to cell ncells - 2
//
// In addition to reduce thread divergence each thread uses a Finite State Machine to decide what value to add to the cell at every step above
// States == 0: targetcell < startcell || targetcell > endcell , contribution = 0;
//           1: targetcell == startcell , contribution = dist in startcell * weight / mu
//           2: targetcell > startcell && targetcell < endcell , contribution = cellsize * weight / mu;
//           3: targetcell == endcell , contribution = dist in endcell * weight / mu
// Transition : 0 --> 1 --> 2 --> 3 --> 0
// Corner Cases : if startcell == endcell transition modified to 0 --> 3 --> 0
//                if startcell == endcell - 1 transition modified to 0 --> 1 --> 3 --> 0
//Thread divergence is reduced since it is possible to calculate how many cells to go before chaning state. So for those many cells, the checking to see which value to add is skipped.

__kernel void gen_particles_tally(__global struct sysdata * data,  __global double* final_tally, __global float4 * RANLUXCLTab, __global int * extra_NPc)
{
  const float cellsize_recip = data->dx_recip;                    //Reciprocal of cellsize
  const float cellsize = data->dx;                                //Cellsize
  const float weight = data->q0_avg;                              //Weight of each particle
  const float weightS2 = weight * weight ;                         
  const float sig_t_recip = data->sig_t_recip;                    //Reciprocal of the total cross section
  const float lx = data->lx;                                      //Size of the domain
  const int numCells = data->nx;                                   //numCells in the domain is same as number of block and number of threads
  const float eps = data->eps;                                    //If mu < eps , replace mu with eps/2
  const float prob_s = data->sig_s * data->sig_t_recip;           //Probability of scattering
  
  const int block_id = get_group_id(0);                            
  const int thread_id = get_local_id(0);
  
  int i,nc;
  int target_cell;

  //Space allocated in shared memory to hold the tally for the current workgroup. 
  __local double group_tally[2*NUM_CELLS];

  //Set group_tally in shared memory to 0
  group_tally[thread_id] = 0.0;
  group_tally[numCells+ thread_id] = 0.0;
  
  //Number of particles to be simulated by this workgroup
  __local int num_particles;
  if( thread_id == 0 )
    num_particles = (int)data->NPc ;
  
  //barrier(CLK_LOCAL_MEM_FENCE);

  //temporary variables
  float4 randoms;
  float start,end,next;
  int startcell,endcell,temp;
  float mu,fx;
  int goneout,state13;
  float startface,endface;
  float all_dist[4];
  float weight_efmu_recip ,weightS2_efmu_recipS2 ;
  int state,nskip,state23,nskip23;
  float dist;
  
  ranluxcl_state_t ranluxclstate;
  //Download the seed for the random number generator
  ranluxcl_download_seed(&ranluxclstate, RANLUXCLTab);
  
  randoms = ranluxcl(&ranluxclstate);                                               //Generate random number
  next = ( block_id + randoms.x ) * cellsize;                                        //Generate starting position of particle
  do{
    randoms = ranluxcl(&ranluxclstate);
    all_dist[0] = 0.0; 
    all_dist[1] = 0.0;
    all_dist[2] = 0.0;

    state13 = 3;                                                                   //By default state 1 is skipped to state 3 for corner case startcell == endcell
    state23 = 2;                                                                   //By default state 2 is not skipped.	These two things should never happen.
    nskip23 = numCells;                                                            //Dummy value, sicne this is updated later

    goneout = 0;                                                                   //Flag to indicate if particle streamed out of system
    mu = -1.0 + 2.0 * randoms.x ;                                                  //mu = cos(polar angle)
    fx = (-log(randoms.y) * sig_t_recip );                                         // Distance travelled along the length of the system
    start = next ;                                                                 //Starting position of the particle
    end = start + mu * fx ;                                                        //Ending position of the particle

    startcell = (int)(start * cellsize_recip);                                     //The cell in which the particle starts
    endcell = (int)(end * cellsize_recip );                                        //Cell in which the particle ends

    //If particle goes out through the left edge of the system
    if( end <= 0.0 )
      {
	end = 0.0;
	endcell = 0;
	goneout = 1;
      }
    //If particle goes out through the right edge
    if( end >= lx )
      {
	end = lx;
	endcell = numCells - 1;
	goneout = 1;
      }	      

    next = end;                                                                    //For the next iteration the particle starts from here
    //Startcell <= endcell, therefore if mu < 0 , switch them 
    if( mu < 0.0 )
      {
	end = start;
	start = next;
	temp = endcell;
	endcell = startcell;
	startcell = temp;
	mu = -mu;
      }
    if( mu < eps )
      mu = eps/2.0;

    weight_efmu_recip = weight / mu ;                                             //Precompute common sub expressions
    weightS2_efmu_recipS2 = weight_efmu_recip * weight_efmu_recip;                //Precompute common sub expressions
		
    all_dist[3] = end - start;                                                    //If startcell == endcell, based on corner case, dist = end - start.
    if( startcell != endcell )
      {
	startface = (startcell+1) * cellsize;
	endface = endcell * cellsize;
	all_dist[1] = startface - start;                                          //Dist travelled in startcell
	all_dist[2] = cellsize;                                                   //Dist travelled in cells from startcell +1 , to endcell -1
	all_dist[3] = end - endface;                                              //Dist travelled in endcell
	state13 = 1;                                                              //Not a case where startcell == endcell, do not skip state 1
	state23 = 2;                                                              //By default assume that startcell != endcell -1 , do not skip state 2
	nskip23 = endcell - startcell -1;                                         //Number of cells to skip before checking for new state
	if( startcell == endcell -1  )                                            //If indeed corner case of startcell == endcell -1
	  {
	    state23 = 3;                                                          //Skip state 2
	    nskip23 = 1;                                                          //Number of cells to skip in the resulting state ( state 3 ) before state change = 1
	  }
      }
		
    //Decide the initial state of the particle
    if( thread_id > endcell )
      {
	state = 0;
	nskip = numCells - thread_id  + startcell;                                //Number of cells to skip before it needs to switch to state 1 
      }
    else if( thread_id == endcell )
      {
	state = 3;
	nskip = 1;                                                                //State3 exists for only 1 cell
      }
    else if( thread_id < endcell && thread_id > startcell )
      {
	state = 2;
	nskip = endcell - thread_id ;                                             //Number of cells to skip before it need to switch to state 3
      }
    else if( thread_id == startcell )
      {
	state = 1;
	nskip = 1;                                                                //state 1 exists for only 1 cell
      }
    else
      {
	state = 0;
	nskip = startcell - thread_id;                                            //Number of cells to skip before it needs to switch to state 1 
      }
      
    //If particle is absorbed or it is gone out, then generate new particle start position
    if( (goneout) || ( !(goneout) && (randoms.z >= prob_s) ) )
      { 
	goneout = 1;
	next = ( block_id + randoms.w ) * cellsize;
      }

    //Tallying statistics of particle streamed by each thread
    for( nc = 0 ; nc < numCells ; nc++ ) 
      {
	target_cell=( thread_id + nc ) % numCells;                                //Calculate cell to contribute to
	dist = all_dist[state];                                                   //Get the dist based on current state

	if( target_cell == 0 )
	  num_particles -= goneout ;                                              //If contributing to cell 0, update the number of particles left to stream ( no itnerference )
	      
	barrier(CLK_LOCAL_MEM_FENCE);

	group_tally[target_cell] += (double)(dist * weight_efmu_recip);                     //Add contribution to phi_n
	group_tally[target_cell+numCells] += (double)(dist* dist * weightS2_efmu_recipS2);  //Add contribution to phi_n2

	nskip--;                                                                  //Reduce number of cells to state change by 1
	//If state needs to change
	if( nskip == 0 )
	  {
	    state = (state+1) & 3;                                                //Next state
	    switch(state){
	    case 0:
	      nskip = numCells - endcell + startcell - 1;                         
	      break;
	    case 1:
	      state = state13;                                                    //Change state based on corner case of startcell == endcell or not
	      nskip = 1;
	      break;
	    case 2:
	      state = state23;                                                    //Change state based on if startcell == endcell -1
	      nskip = nskip23;
	      break;
	    case 3:
	      nskip = 1;
	      break;
	    }
	  }
      }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  //Continue  until all particles of this cell have been streamed
  while( num_particles > 0 );

  //Offset to index into portion of final tally that holds the tally for current workgroup/block
  const int block_offset = block_id * numCells;

  //Offset to index into portion of final tally from where phi_n2 is stored
  const int n2_offset = numCells * numCells;

   //Update tally in coalasced manner
  final_tally[block_offset + thread_id] = group_tally[thread_id];
  final_tally[block_offset + n2_offset + thread_id] = group_tally[numCells+thread_id];
  
  //Update the number of extra particles streamed
  if( thread_id == 0 )
    extra_NPc[block_id] = -num_particles;

  ranluxcl_upload_seed(&ranluxclstate, RANLUXCLTab);
}
  
