The policy predicts wall-reflected catch intervals, anticipating the earliest possible brick bounce while ascending. First-layer units compare paddle position with these intervals and detect motion, bottom-row phase, and edges. Second-layer units gate movement and permit limited catch-safe variation, favoring safer edge bounces.

```wspl
wspl 0.2;
target base=dense(widths=[600,32,32,6],activations=[tanh,tanh,linear],layout=in_out,storage=f32,init=zeros);

const landing_lo=[0,1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1,0,0];
const landing_hi=[1,2,3,4,5,6,7,8,9,9,9,8,7,6,5,4,3,2,1,0];
const down_steps=[8,7,6,5,4,3,2,1,0,0];
const up_steps=[9,8,7,6,5,6,7,8,9,9];
const wall_offset=[0.5,0,0,0,0,0,0,0,0,-0.5];
const vertical_offset=[0.5,0.5,0.5,0.5,0.5,-0.5,-0.5,-0.5,-0.5,-0.5];
const horizontal_sign=[1,-1,1,-1];
const vertical_sign=[1,1,-1,-1];
const action_columns=[1,3];
const steer_sign=[1,1,-1,-1];
const steer_actions=[1,3,1,3];
const steer_gain=[9.7,8.3,8.3,9.7];
const edge_units=[15,15,16,16];
const edge_gain=[1.6,-1.6,-1.6,1.6];

edit base as candidate mode=fixed numeric=exact_to_storage {
  b1[0:8]=-4;
  b1[10]=-8;
  b1[11:15]=-4;
  b1[15]=-60;
  b1[16]=12;

  for rr in range(10) {
    for cc in range(10) {
      W1[(rr*10+cc)*6+3,0]=-8*landing_hi[cc+down_steps[rr]];
      W1[(rr*10+cc)*6+3,1]=8*landing_lo[cc+down_steps[rr]];
      W1[(rr*10+cc)*6+3,2]=-8*(9-landing_lo[9-cc+down_steps[rr]]);
      W1[(rr*10+cc)*6+3,3]=8*(9-landing_hi[9-cc+down_steps[rr]]);

      W1[(rr*10+cc)*6+3,4]=-8*landing_hi[cc+up_steps[rr]];
      W1[(rr*10+cc)*6+3,5]=8*landing_lo[cc+up_steps[rr]];
      W1[(rr*10+cc)*6+3,6]=-8*(9-landing_lo[9-cc+up_steps[rr]]);
      W1[(rr*10+cc)*6+3,7]=8*(9-landing_hi[9-cc+up_steps[rr]]);

      W1[(rr*10+cc)*6+3,8]=8*(cc+wall_offset[cc]);
      W1[(rr*10+cc)*6+4,8]=-8*cc;
      W1[(rr*10+cc)*6+3,9]=8*(rr+vertical_offset[rr]);
      W1[(rr*10+cc)*6+4,9]=-8*rr;

      W1[(rr*10+cc)*6+3,15]=8*cc;
      W1[(rr*10+cc)*6+3,16]=-8*cc;
    }
  }

  for cc in range(10) {
    W1[(90+cc)*6,0:8:2]=8*cc;
    W1[(90+cc)*6,1:8:2]=-8*cc;
    W1[(90+cc)*6,11:15:2]=8*cc;
    W1[(90+cc)*6,12:15:2]=-8*cc;

    W1[(80+cc)*6+3,4:8:2]=-8*(9-cc);
    W1[(80+cc)*6+3,5:8:2]=8*(9-cc);
    W1[(80+cc)*6+3,10]=16;

    W1[(80+cc)*6+3,11]=-8*landing_lo[cc];
    W1[(80+cc)*6+3,12]=8*landing_hi[cc];
    W1[(80+cc)*6+3,13]=-8*(9-landing_hi[9-cc]);
    W1[(80+cc)*6+3,14]=8*(9-landing_lo[9-cc]);
  }

  for branch in range(4) {
    for side in range(2) {
      W2[2*branch+side,2*branch+side]=6;
      W2[8,2*branch+side]=6*horizontal_sign[branch];
      W2[9,2*branch+side]=6*vertical_sign[branch];
      b2[2*branch+side]=-12;
      W3[2*branch+side,action_columns[side]]=18;
    }
  }

  for choice in range(4) {
    W2[11+choice,8+choice]=6;
    W2[8,8+choice]=6*steer_sign[choice];
    W2[9,8+choice]=6;
    W2[10,8+choice]=6;
    b2[8+choice]=-18;
    W3[8+choice,steer_actions[choice]]=steer_gain[choice];

    W2[11+choice,12+choice]=6;
    W2[8,12+choice]=6*steer_sign[choice];
    W2[9,12+choice]=6;
    W2[10,12+choice]=6;
    W2[edge_units[choice],12+choice]=6;
    b2[12+choice]=-24;
    W3[12+choice,steer_actions[choice]]=edge_gain[choice];
  }

  b3[:]=[0,72,-30,72,-30,-30];
}
```