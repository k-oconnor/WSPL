The parent cuts power and stalls on the ramp. I isolate a small feedback controller that backs up at limited speed, then commits to left drive for a momentum-building approach.

```wspl
wspl 0.2;
target base=artifact("parent",sha256="faaafa1c0e3c0d1f3ce2e0de66c72d98621fbaf0a78ec066837af26e6edec56f");
edit base as candidate mode=fixed numeric=native_numpy_v1 {
  W1[0:679,0:2]=0;
  b1[0:2]=[0,10];
  W1[52,0]=1;
  W1[54,1]=1;

  W2[0:128,0:4]=0;
  b2[0:4]=[-52,76,-94,113];
  W2[0,0]=20;
  W2[0,1]=-20;
  W2[1,2]=10;
  W2[1,3]=-10;

  W3[0:128,0:2]=0;
  b3[0:2]=[-12,0];
  W3[0:3,0]=5;
  W3[3,1]=1;

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
  W7[0,0:3]=[18,-18,18];
  W7[1,0:3]=[-6,0,6];
}
```