# Project Worklog

## Commit: setup booksim and run test simulation

### Changes
- setup booksim
- run a test simulation

### How to Test
```bash
git clone git@github.com:booksim/booksim2.git
cd booksim2/src
make clean
make
./booksim ../examples/singleconfig
```
Output:
```
BEGIN Configuration File: ./examples/singleconfig
// $Id$

// Copyright (c) 2007-2015, Trustees of The Leland Stanford Junior University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//A single cross 10X10 cross bar under injection mode

topology = fly;

k = 10;
n = 1;

num_vcs = 8;

vc_buf_size = 8;

vc_allocator = separable_input_first;
sw_allocator = separable_input_first;

routing_function = dest_tag;

traffic = uniform;

use_read_write = 0;

injection_rate = 1.0;




sample_period  = 100000;

routing_delay  = 0;
vc_alloc_delay = 1;
sw_alloc_delay = 1;
st_final_delay = 1;

END Configuration File: ./examples/singleconfig
Class 0:
Packet latency average = 18924.1
        minimum = 6
        maximum = 38073
Network latency average = 103.062
        minimum = 6
        maximum = 728
Slowest packet = 2025
Flit latency average = 103.062
        minimum = 6
        maximum = 728
Slowest flit = 107831
Fragmentation average = 0
        minimum = 0
        maximum = 0
Injected packet rate average = 0.626668
        minimum = 0.62068 (at node 8)
        maximum = 0.63559 (at node 7)
Accepted packet rate average = 0.626022
        minimum = 0.62063 (at node 2)
        maximum = 0.63081 (at node 7)
Injected flit rate average = 0.626668
        minimum = 0.62068 (at node 8)
        maximum = 0.63559 (at node 7)
Accepted flit rate average= 0.626022
        minimum = 0.62063 (at node 2)
        maximum = 0.63081 (at node 7)
Injected packet length average = 1
Accepted packet length average = 1
Total in-flight flits = 650 (0 measured)
latency change    = 1
throughput change = 1
Average latency for class 0 exceeded 500 cycles. Aborting simulation.
Too many sample periods needed to converge
Simulation unstable, ending ...
Total run time 4.27995
```
---

## Template

Use the template below for new entries:

```markdown
## Commit: [message]

### Changes
- change 1
- change 2

### How to Test
```bash
# Test steps here
```

### Notes
- Additional information
```
