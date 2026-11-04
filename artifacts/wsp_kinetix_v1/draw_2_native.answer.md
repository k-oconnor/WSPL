The parent stalls at the crest while alternating drive and coast. This candidate backs up slowly for a longer run-up, then drives left, with pitch guards to limit flipping. Remaining network channels are retained.

```edits
# Isolated position, velocity, and projected-pitch features.
embed.W[0:679,118:128] = 0
embed.b[118:128] = 0
embed.W[52,118] = 1
embed.W[54,119] = 0.3
embed.W[236,119] = 0.35
embed.W[254,119] = 0.35
embed.b[119] = 10
embed.W[64,120] = 1
embed.W[59,120] = 1.5
embed.b[120] = 3

# Position, direction, backing-speed, and pitch comparisons.
actor0.W[0:128,118:128] = 0
actor0.W[118:128,0:118] = 0
actor0.b[118:128] = [77,-195,-26,220,61,-28,-26,-17.5,34,0]
actor0.W[118,118] = -20
actor0.W[119,119] = 20
actor0.W[118,120] = 20
actor0.W[119,121] = -20
actor0.W[118,122] = -20
actor0.W[118,123] = 20
actor0.W[120,124] = 10
actor0.W[120,125] = 10
actor0.W[120,126] = -10

# Backing phase, ramp region, and guard signals.
actor1.W[0:128,118:128] = 0
actor1.W[118:128,0:118] = 0
actor1.b[118:128] = [-12.5,-7.5,0,0,0,0,0,0,0,0]
actor1.W[118:121,118] = 5
actor1.W[122:124,119] = 5
actor1.W[121,120] = 5
actor1.W[124,121] = 5
actor1.W[125,122] = 5
actor1.W[126,123] = 5

# Back slowly; otherwise drive left with terrain-dependent pitch limits.
actor2.W[0:128,118:128] = 0
actor2.W[118:128,0:118] = 0
actor2.b[118:128] = [-12.5,-7.5,-12.5,0,0,0,0,0,0,0]
actor2.W[118,118] = 5
actor2.W[120,118] = 5
actor2.W[123,118] = 5
actor2.W[118,119] = -5
actor2.W[121,119] = 5
actor2.W[118,120] = -5
actor2.W[119,120] = 5
actor2.W[122,120] = 5

# Combine left-drive branches and sharpen decisions.
actor3.W[0:128,118:128] = 0
actor3.W[118:128,0:118] = 0
actor3.b[118:128] = [0,2.5,0,0,0,0,0,0,0,0]
actor3.W[118,118] = 5
actor3.W[119:121,119] = 5

actor4.W[0:128,118:128] = 0
actor4.W[118:128,0:118] = 0
actor4.b[118:128] = 0
actor4.W[118,118] = 5
actor4.W[119,119] = 5

# Override only the effective motor binding; coast when neither drive is safe.
actor_head.W[0:128,0:3] = 0
actor_head.W[118:128,3:16] = 0
actor_head.b[0:3] = 0
actor_head.W[119,1] = 12
actor_head.W[118,2] = 12
```