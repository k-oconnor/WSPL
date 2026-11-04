The policy estimates a wall-reflected downward intercept, preferring the pre-impact column but accepting either catch column near the bottom. Separate units recognize paddle bounces and anticipate the outgoing direction. First-layer units encode direction, interception comparisons, and phase; second-layer units select movement, with unused units left zero.

```wspl
wspl 0.2;
target base=dense(widths=[600,32,32,6],activations=[tanh,tanh,linear],layout=in_out,storage=f32,init=zeros);

const dir_ball=[0,0,2,2,4,4,6,6,8,8];
const dir_trail=[1,-1,-1,-3,-3,-5,-5,-7,-7,-9];
const edge_bias=[0.5,0,0,0,0,0,0,0,0,-0.5];
const paddle_sign=[-1,1,-1,1];

const right_target=[
  [8,9,9,8,7,6,5,4,3,2],
  [7,8,9,9,8,7,6,5,4,3],
  [6,7,8,9,9,8,7,6,5,4],
  [5,6,7,8,9,9,8,7,6,5],
  [4,5,6,7,8,9,9,8,7,6],
  [3,4,5,6,7,8,9,9,8,7],
  [2,3,4,5,6,7,8,9,9,8],
  [1,2,3,4,5,6,7,8,9,9],
  [0,1,2,3,4,5,6,7,8,9],
  [0,1,2,3,4,5,6,7,8,9]
];

const near_low=[
  [1,2,3,4,5,6,7,8,9,8],
  [0,1,2,3,4,5,6,7,8,9]
];

const near_high=[
  [2,3,4,5,6,7,8,9,9,9],
  [1,2,3,4,5,6,7,8,9,9]
];

edit base as candidate mode=fixed numeric=exact_to_storage {
  b1[0:4]=-75;
  b1[4]=-9;
  b1[5:7]=-288;
  b1[7]=-33;

  for row in range(10) {
    for col in range(10) {
      W1[(row*10+col)*6+3,0]=6*(right_target[row,col]+12*dir_ball[col]);
      W1[(row*10+col)*6+3,1]=6*(-right_target[row,col]+12*dir_ball[col]);
      W1[(row*10+col)*6+3,2]=6*(9-right_target[row,9-col]-12*dir_ball[col]);
      W1[(row*10+col)*6+3,3]=6*(-9+right_target[row,9-col]-12*dir_ball[col]);

      W1[(row*10+col)*6+4,0:2]=72*dir_trail[col];
      W1[(row*10+col)*6+4,2:4]=-72*dir_trail[col];

      W1[(row*10+col)*6+3,5]=6*(col+edge_bias[col]);
      W1[(row*10+col)*6+3,6]=-6*(col+edge_bias[col]);
      W1[(row*10+col)*6+4,5]=6*col;
      W1[(row*10+col)*6+4,6]=-6*col;

      W1[(row*10+col)*6+3,7]=6*row;
      W1[(row*10+col)*6+4,7]=-6*row;
    }
  }

  for col in range(10) {
    W1[540+6*col,0:4]=6*col*paddle_sign;
    W1[540+6*col,5]=-12*col;
    W1[540+6*col,6]=12*col;
  }

  W1[483:540:6,4]=6;
  W1[484:540:6,4]=6;
  W1[483:540:6,5:7]+=144;
  W1[484:540:6,5:7]+=144;
  W1[423:540:6,7]+=30;

  W1[:,8:12]=copy(W1[:,0:4]);
  b1[8:12]=copy(b1[0:4]);

  for band in range(2) {
    for col in range(10) {
      W1[((band+7)*10+col)*6+3,8]=6*(near_low[band,col]+12*dir_ball[col]);
      W1[((band+7)*10+col)*6+3,9]=6*(-near_high[band,col]+12*dir_ball[col]);
      W1[((band+7)*10+col)*6+3,10]=6*(9-near_high[band,9-col]-12*dir_ball[col]);
      W1[((band+7)*10+col)*6+3,11]=6*(-9+near_low[band,9-col]-12*dir_ball[col]);
    }
  }

  W2[0,0]=4;
  W2[2,0]=4;
  W2[1,1]=4;
  W2[3,1]=4;

  W2[8,2]=4;
  W2[10,2]=4;
  W2[9,3]=4;
  W2[11,3]=4;

  W2[4,0:4]=-4;
  W2[7,0:2]=-4;
  W2[7,2:4]=4;
  b2[0:4]=-4;

  W2[5,4]=4;
  W2[6,5]=4;

  W3[0:6:2,3]=16;
  W3[1:6:2,1]=16;
  b3[1]=32;
  b3[3]=32;
  b3[2]=-32;
  b3[4:6]=-32;
}
```