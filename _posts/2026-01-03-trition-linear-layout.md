---
layout: post
title:  "Notes on Linear Layout in Triton"
date: 2026-01-03 12:00:00 +0000
categories: update
---
sources: 

- Paper: https://arxiv.org/pdf/2505.23819
- Awesome blog post: https://www.lei.chat/posts/triton-linear-layout-concept/
-we aremplementation Header: https://github.com/triton-lang/triton/blob/d9facf3a6edbc819c80d58b87e689bc0c2632756/include/triton/Tools/LinearLayout.h

In the following tries to build some intuition behind the Linear Layouts in triton by "Rediscovering" the foundation of LLs with an example.

For this assume we are working with a imaginative GPU whose warpsize is 4.

Lets assume we want to load this 8x4 matrix with our GPU into registers.

The 8x4 matrix is displayed here. Every value at (row,col) is the linear offset into memory, so for example accessing (3,1) we would have to access memory at offset 13.

```
          col →
            0    1    2    3
        ------------------------
row   0 |   0    1    2    3
row   1 |   4    5    6    7
row   2 |   8    9   10   11
row   3 |  12   13   14   15
row   4 |  16   17   18   19
row   5 |  20   21   22   23
row   6 |  24   25   26   27
row   7 |  28   29   30   31
```

Now if we would want to load this matrix with our GPU using a single CTA of 2 warps with 4 threads each we could distribute the matrix across the warps/lanes/register in the following way:

We denote the ownership as: `(wXtYrZ) = (warp X | lane Y | register Z)`

```
          col →
            0           1           2           3
row 0    0:w0l0r0    1:w0l0r1    2:w0l1r0    3:w0l1r1
row 1    4:w0l0r2    5:w0l0r3    6:w0l1r2    7:w0l1r3
row 2    8:w0l2r0    9:w0l2r1   10:w0l3r0   11:w0l3r1
row 3   12:w0l2r2   13:w0l2r3   14:w0l3r2   15:w0l3r3
      ------------------------------------------------
row 4   16:w1l0r0   17:w1l0r1   18:w1l1r0   19:w1l1r1
row 5   20:w1l0r2   21:w1l0r3   22:w1l1r2   23:w1l1r3
row 6   24:w1l2r0   25:w1l2r1   26:w1l3r0   27:w1l3r1
row 7   28:w1l2r2   29:w1l2r3   30:w1l3r2   31:w1l3r3

```


Typically when writing a GPU kernel you would calculate the row and col index for every thread like in the following:

```
int thread_id = (id from 0...7)
int warp_id = thread_id / 4; (id from 0..1)
int lane_id = thread_id % 4; (id from 0..3)
int reg_id = (id from 0..3)

int w0 = warp_id

int l0 = lane_id / 2
int l1 = lane_id %  2

int r0 = reg_id / 2
int r1 = reg_id % 2

int row = w0 * 4 + l0 * 2 + r0
int col = l1 * 2 + r1
```

Lets take a look at the same example from above but in binary.

```
          col →
          00        01        10        11
row 000 | 0b00000  0b00001  0b00010  0b00011
row 001 | 0b00100  0b00101  0b00110  0b00111
row 010 | 0b01000  0b01001  0b01010  0b01011
row 011 | 0b01100  0b01101  0b01110  0b01111
row 100 | 0b10000  0b10001  0b10010  0b10011
row 101 | 0b10100  0b10101  0b10110  0b10111
row 110 | 0b11000  0b11001  0b11010  0b11011
row 111 | 0b11100  0b11101  0b11110  0b11111
```

We can make some cool observations here:

(NOTE: bit indices in binary are usually given from right to left, which is a convention i will also follow here!)

- the least significant bits in idx 0, 1 fully describe which column we are indexing
- the most significant bits at idx 2, 3, 4 fully describe which row we are indexing

If we apply the same warp/lane/register distribution as above there are some more observations we can make:

```
          00                            01                            10                            11
row 000 | 0b00000:w0b0l0b00r0b00        0b00001:w0b0l0b00r0b01        0b00010:w0b0l0b01r0b00        0b00011:w0b0l0b01r0b01
row 001 | 0b00100:w0b0l0b00r0b10        0b00101:w0b0l0b00r0b11        0b00110:w0b0l0b01r0b10        0b00111:w0b0l0b01r0b11
row 010 | 0b01000:w0b0l0b10r0b00        0b01001:w0b0l0b10r0b01        0b01010:w0b0l0b11r0b00        0b01011:w0b0l0b11r0b01
row 011 | 0b01100:w0b0l0b10r0b10        0b01101:w0b0l0b10r0b11        0b01110:w0b0l0b11r0b10        0b01111:w0b0l0b11r0b11
      -----------------------------------------------------------------------------------------------------------------------------
row 100 | 0b10000:w0b1l0b00r0b00        0b10001:w0b1l0b00r0b01        0b10010:w0b1l0b01r0b00        0b10011:w0b1l0b01r0b01
row 101 | 0b10100:w0b1l0b00r0b10        0b10101:w0b1l0b00r0b11        0b10110:w0b1l0b01r0b10        0b10111:w0b1l0b01r0b11
row 110 | 0b11000:w0b1l0b10r0b00        0b11001:w0b1l0b10r0b01        0b11010:w0b1l0b11r0b00        0b11011:w0b1l0b11r0b01
row 111 | 0b11100:w0b1l0b10r0b10        0b11101:w0b1l0b10r0b11        0b11110:w0b1l0b11r0b10        0b11111:w0b1l0b11r0b11

```

The key observation we can make here is that each bit of the warp/lane/reg ID controls a unique bit of the output offset. 

Lets first only take a look at the register IDs.

- We can see that if the bit at idx 0 of the register ID is 0 the offset bit at idx 0 is also 0. Consequently if bit 0 of the register ID is 1, bit 0 of the offset bit is also 1. 
- We can also see that if the bit at idx 1 of the register ID is 0 the offset bit at idx 2 is 0. Consequently if bit 1 of the register ID is 1, bit 2 of the offset bit is also 1. 

Combinding this with the observation above means the register bits control one row bit and one column bit. 

We can make the same observation for the thread and warp IDs as well. So we would end up with the following ownership mapping:

```
reg_id    offset
0b01    | 0b00001
0b10    | 0b00100

lane_id | offset
0b01    | 0b00010
0b10    | 0b01000

warp    | offset
0b1     | 0b10000
```


We can use this to build a generic ND to MD index mapping system (Linear Layouts).

Lets say we now want to map the 3D (reg,lane,warp) dimension tuple to the 2D (rows,cols) of our output matrix.

We only have to be aware which bits contribute "how much" to which output dimension, for example:

```
reg  = 1 (0b01) -> col += 1
reg  = 2 (0b10) -> row += 1
lane = 1 (0b01) -> col += 2 
lane = 2 (0b10) -> row += 2
warp = 1 (0b1)  -> row += 4
```

Now similarly we could also map directly to the 1D output offset:

```
reg  = 1 (0b01) -> offset += 1
reg  = 2 (0b10) -> offset += 4
lane = 1 (0b01) -> offset += 2 
lane = 2 (0b10) -> offset += 8
warp = 1 (0b1)  -> offset += 16
```

Turns out the bit ownership we described above directly translate into the "A" matrix from the paper. 

The only thing we have to do is convert the offset incremets into binary and concatenate them into a matrix. 

(Turns out this is also what is referred to as the basis of the linear layout!)

```
Basis vectors:           1     4      2      8     16
In binary:           00001 00100  00010  01000  10000
```

Write each basis vector as a column (LSB at top):

```
             Reg       Lane     Warp
          bit0 bit1  bit0 bit1  bit0
    A =   [ 1    0     0    0     0 ]
          [ 0    0     1    0     0 ]
          [ 0    1     0    0     0 ]
          [ 0    0     0    1     0 ]  
          [ 0    0     0    0     1 ]
```

Using this information, we could now implement the reg/lane/warp index mapping the following way:

```
int thread_id = (id from 0...7)
int warp_id = thread_id / 4; (id from 0..1)
int lane_id = thread_id % 4; (id from 0..3)
int reg_id = (id from 0..3)

int offset = 0;

int r0 = (reg_id & 1)        # r0 owns bit 0 (don't have to shift here and could also precomute this at compile time)
int r1 = (reg_id & 2)        # r1 owns bit 2 (same here)
int l0 = (lane_id & 1) << 1  # l0 owns bit 1
int l1 = (lane_id & 2) << 2  # l1 owns bit 3 (we only shift by two because its already at idx 1 and not 0)
int w0 = (warp_id & 1) << 4  # w0 owns bit 4

int offset = r0 | r1 | l0 | l1 | w0
```

This would basically be the code the `apply` of a Linear Layout would generate. 

