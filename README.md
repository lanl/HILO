HILO: Quasi Diffusion Accelerated Monte Carlo on Hybrid Architectures
===========

Abstract
-----------

The Boltzmann transport equation provides high fidelity simulation of a
diverse range of kinetic systems. Classical methods to solve
the equation are computationally and data intensive. Existing stochastic
solutions to the Boltzmann equation map well to traditional large
multi-core and
many-node architectures but suffer performance degradations on graphics
processing units (GPUs) due to heavy thread divergence. We present a
a novel algorithm, Quasi-Diffusion Accelerated Monte Carlo (QDA-MC),
which improves performance on heterogeneous CPU/GPU architectures.

An equally important aspect of this project is the joint development
of QDA-MC through collaboration between the computational and computer
science communities. This collaboration identified computational platforms
and features that best suit the algorithm, and influenced algorithmic details
which improve its computational efficiency. In addition to algorithm details
and implementation results, we present the code optimizations and the design
decisions that were critical to the co-design process.

License
-------

This code is released under LA-CC-11-076. The license is
BSD-ish with a "modifications must be indicated" clause.  See
<http://github.com/losalamos/HILO/blob/master/LICENSE> for the full
text.


Authors
-------

Student authors developed code while interns at Los Alamos during the Summer 2011 Co-Design School (http://codesign.lanl.gov)

### Students
Mahesh Ravishankar ravishan@cse.ohio-state.edu   
Jeffrey Willert jawiller@ncsu.edu   
Paul Sathre sath6220@cs.vt.edu   
Han Dong handong32@gmail.com   
Michael Sullivan mbsullivan@mail.utexas.edu   
William Taitano williamtaitano1208@hotmail.com   

### Los Alamos Mentors
Tim Germann   
Dana Knoll   
Bryan Lally   
Patrick McCormick   
Allen McPherson   
Scott Pakin   