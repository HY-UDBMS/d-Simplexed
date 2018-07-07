# d-Simpled Framework in d-dimensions (f_1,f_2,...,f_d) => runtime

A work in progress.  The d-Simplexed framework can be extended to an arbitrary number of dimensions.  To extend to >4 dimensions, copy delaunaymodel.py and sampler.py from ../3d or ../4d and modify accordingly.

Ideally when this is written, it could replace ../3d and ../4d, however the logic is much more complex to write and debug, hence why we split 3d and 4d up separately, since there are specific methods that we can apply in 3d and 4d to make things easier that aren't extensible to higher dimensions.
