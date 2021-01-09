# Julia/Clojure KMeans-MNist

An example of using using Clojure and Julia to implement KMeans to 
create an mnist classifier which performs extremely well and does 
surprisingly well on mnist.


## Usage

If you are on Ubuntu and using JDK 8, type `source scripts/local-julia-env` 
to setup the julia environment.  If you are not on Ubuntu, then see the script
and change/enhance it to fit your system.

From a julia prompt, install the StaticArrays package:

```julia
julia> import(Pkg)
julia> Pkg.add("StaticArrays")
```

In the script you will notice that it exports LD_PRELOAD so that
a special java library is loaded: libjsig.so.  This allows the JVM to forward
signals meant for Julia to Julia but still use signals.  This is a requirement
to run the demonstration.


Assuming that you have the proper environment setup, go into the mnist
namespace and type:

```clojure
kmeans-mnist.mnist> (def data (time (test-kmeans 100)))
...(lots of logging)...
"Elapsed time: 6936.24221 msecs"
#'kmeans-mnist.mnist/data
kmeans-mnist.mnist> data
{:accuracy 0.9577, :confusion-matrix #tech.v3.tensor<int64>[10 10]
[[1932    1    9    2    1    5   12    2    7    3]
 [   1 2262    8    0    3    2    4   13    5    3]
 [   9    8 1964    8    2    2    5   21   19    4]
 [   2    0    8 1904    1   35    0    6   28   20]
 [   1    3    2    1 1864    4    5    9    7   53]
 [   5    2    2   35    4 1684   10    2   28   14]
 [  12    4    5    0    5   10 1884    0    4    1]
 [   2   13   21    6    9    2    0 1948    7   39]
 [   7    5   19   28    7   28    4    7 1824    9]
 [   3    3    4   20   53   14    1   39    9 1888]]}
```


## Some Highlights

The algorithm is a modified kmeans-++ where we use a cumulative summation
for the `++` part as opposed to a sort operation.  Next we calculate distances
inline with finding the centroid index thus reducing algorithm steps.

The kmeans.jl file is somewhat type hinted to allow the Julia compiler as much
freedom as possible to specifically optimize the code to the datatypes of the
arguments provided as well as the number of columns.

I implemented a Julia threading primitive, indexed_map_reduce that allows me 
to efficiently declare context local to a thread before iterating through the
indexes of the parallelization.  This allows a very efficient multithreaded form
of reduction.


* [Externally defined Julia code](resources/kmeans.jl)
* [Calling Julia like Clojure](src/kmeans_mnist/jl_kmeans.clj)
* [Simple, efficient mnist loader](src/kmeans_mnist/mnist.clj)
