#include "RANLUXCL.cl"

struct sysdata {
    float sig_t;
    float dx;
};

__kernel void Kernel_RANLUXCL_Warmup(__global float4* RANLUXCLTab) {
    //Downloading RANLUXCLTab. The state of RANLUXCL is stored in ranluxclstate.
    ranluxcl_state_t ranluxclstate;
    ranluxcl_download_seed(&ranluxclstate, RANLUXCLTab);

    ranluxcl_warmup(&ranluxclstate);

    //Uploading RANLUXCLTab
    ranluxcl_upload_seed(&ranluxclstate, RANLUXCLTab);
}

__kernel void particleGeneration(__constant struct sysdata * data, __global float * mu0, __global float * x0, __global float * xf, __global float4* RANLUXCLTab, const unsigned int cellid, const unsigned int offset, const unsigned int iter)
{
	float fx;
	int i, globalsize, cid, pSpace;
	
	float reuse[3]; 
	
	float4 randoms;
	        
	//get thread/particle id
	int tid = get_global_id(0);
	globalsize = get_global_size(0);
	
	//set up random number generator state
	ranluxcl_state_t ranluxclstate;
    ranluxcl_download_seed(&ranluxclstate, RANLUXCLTab);
    
    for(i=iter-1; i >=0; i--)
    {
    	pSpace = tid+i*globalsize;
    	cid = pSpace / cellid;
    	
    	//randoms = ranluxcl(&ranluxclstate);
    	
    	if(i&3)
    	{
    		randoms = ranluxcl(&ranluxclstate);
    		reuse[(i&3)-1] = randoms.w;
    	}
    	else
    	{
    		randoms = (float4) { reuse[0], reuse[1], reuse[2], 0.0f };
    	}
		
    	//generates particles
		randoms.x = (randoms.x * 2.0f) - 1.0f;
		mu0[pSpace+offset] = randoms.x;
		
		fx = (-log(randoms.y) / data->sig_t);
		
		randoms.z = (cid + randoms.z) * data->dx;
		x0[pSpace+offset] = randoms.z;
		
		xf[pSpace+offset] = randoms.z + randoms.x * fx;	
    }
    ranluxcl_upload_seed(&ranluxclstate, RANLUXCLTab);

}
