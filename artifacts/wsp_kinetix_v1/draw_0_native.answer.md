The parent loses momentum by alternating drive and coast on the ramp. This gives binding 0 a run-up controller: back away, reverse into sustained leftward drive, then coast after clearing the red strip.

```edits
embed.W[0:679, 0:3] = 0
embed.b[0:3] = 0
embed.W[52, 0] = 1
embed.W[54, 1] = -1
embed.W[234, 2] = 1

actor0.W[0:128, 0:4] = 0
actor0.b[0:4] = [-340, 265, -50, 95]
actor0.W[0, 0] = 100
actor0.W[0, 1] = -100
actor0.W[1, 2] = 100
actor0.W[2, 3] = -100

actor1.W[0:128, 0:2] = 0
actor1.b[0:2] = [20, 0]
actor1.W[0:3, 0] = 10
actor1.W[3, 1] = 3

actor2.W[0:128, 0:2] = 0
actor2.b[0:2] = 0
actor2.W[0, 0] = 3
actor2.W[1, 1] = 3

actor3.W[0:128, 0:2] = 0
actor3.b[0:2] = 0
actor3.W[0, 0] = 3
actor3.W[1, 1] = 3

actor4.W[0:128, 0:2] = 0
actor4.b[0:2] = 0
actor4.W[0, 0] = 3
actor4.W[1, 1] = 3

actor_head.W[0:128, 0:3] = 0
actor_head.b[0:3] = 0
actor_head.W[0, 1:3] = [15, -15]
actor_head.W[1, 0] = 40
```