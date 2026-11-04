The parent stalls in an off/forward cycle on the ramp. This candidate first settles, backs up for a run-up, then uses position and velocity to maintain leftward drive through the jump. Most weights and inactive action heads remain unchanged.

```wspl
wspl 0.2;
target base=artifact("parent",sha256="faaafa1c0e3c0d1f3ce2e0de66c72d98621fbaf0a78ec066837af26e6edec56f");
edit base as candidate mode=fixed numeric=native_numpy_v1 {
  W1[0:679,0:3]=0;
  b1[0:3]=[0,10,0];
  W1[52,0]=1;
  W1[54,1]=-1;
  W1[253,2]=1;

  W2[0:128,0:8]=0;
  b2[0:8]=[-177.5,132.5,-525,-54.4,240,-216,-487.5,512.5];
  W2[0,0:8]=[50,-50,0,0,-80,80,0,0];
  W2[1,0:8]=[0,0,50,0,0,0,50,-50];
  W2[2,3]=80;

  W3[0:128,0:2]=0;
  b3[0:2]=[10,-20];
  W3[0:3,0]=5;
  W3[3:8,1]=5;

  W4[0:128,0:2]=0;
  b4[0:2]=0;
  W4[0,0]=3;
  W4[1,1]=3;

  W5[0:128,0:2]=0;
  b5[0:2]=0;
  W5[0,0]=3;
  W5[1,1]=3;

  W6[0:128,0:2]=0;
  b6[0:2]=0;
  W6[0,0]=3;
  W6[1,1]=3;

  W7[0:128,0:3]=0;
  b7[0:3]=0;
  W7[0,1]=12;
  W7[0,2]=-12;
  W7[1,0]=28;
}
```