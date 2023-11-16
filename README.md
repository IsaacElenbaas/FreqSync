# FreqSync
## About
This is probably the worst code I have ever written. Not even in a "it spiralled out of control" way, it was horrible from start to finish. That's what I get for writing it all at 3am. This README is terrible to match the code.
It works, though, which is nice - this was just meant to be a proof-of-concept. Unfortunately the frequency shifting causes more deadzones than I expected and introduces crackling.
The only dependency is ffmpeg, throw some stuff through it if you want.

FreqSync works best with equal-temperament instrumental tracks that do not contain string instruments. This is because it determines a key base and groups all frequencies into notes; non-equal temperament tracks or ones cluttered with more-in-tune-than-equal vocals or strings will result in incorrect final note positions.

### Thoughts
My original plan was to try to correct equal temperament to justified temperament. That changed, this demo tries to maximize the chance for harmonic frequency ratios by shifting the notes within an octave. That ends up creating a unique scale per song, and from the things I tested it on I had about a 50/50 success rate - but when it did succeed, it was generally a notable improvement.

The second original plan was to not group anything and determine the average error (distance from a harmonic for each other frequency scaled by how often they are used) for each frequency as it went up and down within a limited range. I would then. . . somehow. . . apply my sick least-squared-error function I made for [orbits' camera](https://github.com/IsaacElenbaas/orbits/blob/master/orbits.pde#L93-L105) and have used in other projects to create a series of transformations.

That was too much work for a proof of concept so I went with this. FreqSync finds what is maybe a key base and groups frequencies into notes. Each note is given an associativity with each other based on how often and how loudly frequencies grouped to them are played together. Then error, as above defined as distance to the closest note position that encourages harmonic frequency ratios between the two notes, is brute-force minimized.

End terrible README, I'm probably never touching this project again. It did what I wanted it to (plus some crackling), I count it as a success.
