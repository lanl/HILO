/* 
 * CMC Gen and Tally Macros
 */

#ifndef __CMC_GEN_TALLY__
#define __CMC_GEN_TALLY_H__

#define MU_FX \
  { \
    mu = unirandMT_0_1(&dsfmt); \
    fx = (-log(unirandMT_0_1(&dsfmt)) * sig_t_recip); \
  }

#define PRECOMPUTES \
  { \
		/* put mu in a nicer format (reciprocal and respecting epsilon boundaries*/ \
		efmu = (mu > eps) ? mu : eps/2;					  			/* with epsilon boundaries */ \
		efmu_recip = 1.0 / efmu;												/* reciprocal */ \
		efmu_recipS2 = efmu_recip*efmu_recip;						/* for STDEV */ \
		\
		/* pre-compute repeatedly used measurements*/ \
		weight_efmu_recip = weight*efmu_recip; \
		weight_efmu = weight*efmu; \
		weight_cellsize = weight*cellsize; \
		weightS2_efmu_recipS2 = weightS2*efmu_recipS2; \
  }

#define CORNER_CELLS \
  { \
		/* tally corner cells */ \
		if (startcell == endcell){ \
			/* tally once, with difference of particles */ \
			absdist = (end-start); \
			tallies->phi_n[startcell] += absdist * weight_efmu_recip; \
		  tallies->phi_n2[startcell] += absdist * absdist * weightS2_efmu_recipS2; \
		  tallies->E_n[startcell] += absdist * weight_efmu; \
		} \
 		else{ \
			/* starting cell */ \
			begin_rightface = (startcell + 1) * cellsize; \
			absdist = fabs(begin_rightface - start); /* otherwise -0.0f can mess things up */ \
			assert(absdist >= 0); \
			tallies->phi_n[startcell] += absdist * weight_efmu_recip; \
		  tallies->phi_n2[startcell] += absdist * absdist * weightS2_efmu_recipS2; \
		  tallies->E_n[startcell] += absdist * weight_efmu; \
			/* ending cell */ \
			end_leftface = endcell * cellsize; \
			absdist = fabs(end - end_leftface);	/* otherwise -0.0f can mess things up */ \
			assert(absdist >= 0); \
			tallies->phi_n[endcell] += absdist * weight_efmu_recip; \
		  tallies->phi_n2[endcell] += absdist * absdist * weightS2_efmu_recipS2; \
		  tallies->E_n[endcell] += absdist * weight_efmu; \
		} \
  }

#define INNER_LOOP_PRECOMPUTES \
  { \
		weight_cellsize_efmu_recip = (afloat)weight_efmu_recip*cellsize; \
	  weightS2_cellsizeS2_efmu_recipS2 = (afloat)weightS2_efmu_recipS2*cellsizeS2; \
	  weight_cellsize_efmu = (afloat)weight_efmu*cellsize; \
		/*weight_cellsize_efmu_recip = (afloat)weight*efmu_recip*cellsize; */ \
	  /*weightS2_cellsizeS2_efmu_recipS2 = (afloat)weightS2*efmu_recipS2*cellsize*cellsize; */ \
	  /*weight_cellsize_efmu = (afloat)weight*efmu*cellsize; */ \
  }

#define INNER_LOOP \
  { \
    for (k = startcell+1; k <= endcell-1; k++) { \
			tallies->phi_n[k] += weight_cellsize_efmu_recip; \
			tallies->phi_n2[k] += weightS2_cellsizeS2_efmu_recipS2; \
			tallies->E_n[k] += weight_cellsize_efmu; \
    } \
  }




#define INNER_SSE \
  { \
    startcell++; endcell--; \
    int start_unaligned =  ( startcell ) & 0x01; \
    int end_unaligned = ( endcell ) & 0x01; \
    tallies->phi_n[startcell] += weight_cellsize_efmu_recip * start_unaligned; \
    tallies->phi_n2[startcell] += weightS2_cellsizeS2_efmu_recipS2 * start_unaligned; \
    tallies->E_n[startcell] += weight_cellsize_efmu * start_unaligned; \
    \
    register __m128d phi_n_value_add = _mm_set1_pd(weight_cellsize_efmu_recip); \
    register __m128d phi_n2_value_add = _mm_set1_pd(weightS2_cellsizeS2_efmu_recipS2); \
    register __m128d E_n_value_add = _mm_set1_pd( weight_cellsize_efmu); \
    int aligned_start = startcell + start_unaligned; \
    int aligned_end = endcell - end_unaligned; \
    register __m128d phi_n_value_acc; \
    register __m128d phi_n2_value_acc; \
    register __m128d E_n_value_acc; \
    for (k = aligned_start; k < aligned_end; k+=VECTOR_WIDTH) \
      { \
	phi_n_value_acc = _mm_load_pd(tallies->phi_n+k); \
	phi_n2_value_acc = _mm_load_pd(tallies->phi_n2+k); \
	E_n_value_acc = _mm_load_pd(tallies->E_n+k); \
	phi_n_value_acc = _mm_add_pd(phi_n_value_acc,phi_n_value_add); \
	phi_n2_value_acc = _mm_add_pd(phi_n2_value_acc,phi_n2_value_add); \
	E_n_value_acc = _mm_add_pd(E_n_value_acc,E_n_value_add); \
	_mm_store_pd(tallies->phi_n+k,phi_n_value_acc); \
	_mm_store_pd(tallies->phi_n2+k,phi_n2_value_acc); \
	_mm_store_pd(tallies->E_n+k,E_n_value_acc); \
      }						    \
    \
    tallies->phi_n[endcell] += weight_cellsize_efmu_recip  * end_unaligned ;  \
    tallies->phi_n2[endcell] += weightS2_cellsizeS2_efmu_recipS2 * end_unaligned ; \
    tallies->E_n[endcell] += weight_cellsize_efmu * end_unaligned ; \
  } 


#define SANITY_CHECKS\
  { \
			assert(startcell <= endcell); \
			assert(startcell >= 0 && endcell <= numCells - 1); \
			assert(start <= end); \
			assert(start >= 0.0); \
			assert(end <= lx); \
  }

#endif
