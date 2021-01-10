using StaticArrays

"""
A simple, high performance method of implemention a multithreaded map/reduce
algorithm.  index_map_fn is called nthreads times and takes 2 integers, 
start_idx and group_len.   The results of the nthreads invocations
are all then passed at once into reduce_fn.
"""
function indexed_map_reduce(num_iters::Int,
                            index_map_fn,
                            reduce_fn=nothing)
    parallelism = Threads.nthreads()
    if (num_iters < (2 * parallelism)
        || ccall(:jl_in_threaded_region, Cint, ()) != 0
        || parallelism <= 1)
        int_data = index_map_fn(1, num_iters)
        if reduce_fn != nothing
            reduce_fn([int_data])
        end
    else
        ccall(:jl_enter_threaded_region, Cvoid, ())
        tasks = Vector{Task}(undef, parallelism)
        try
            group_len, rem = divrem(num_iters, parallelism)
            for idx = 1:parallelism
                didx = idx - 1
                any_leftover = didx < rem
                # back to 1 based
                start_idx = didx * group_len + 1
                llen = group_len
                if any_leftover
                    start_idx += didx
                    llen += 1
                else
                    start_idx += rem
                end
                t = Task(function() index_map_fn(start_idx, llen) end)
                t.sticky = true
                ccall(:jl_set_task_tid, Cvoid, (Any, Cint), t, idx-1)
                tasks[idx] = t
                schedule(t)
            end
        finally
            ccall(:jl_exit_threaded_region, Cvoid, ())
        end
        if (reduce_fn != nothing)
            # Perhaps type the intermediate result
            results = Vector{Any}(undef, parallelism)
            for idx = 1:parallelism
                results[idx] = fetch(tasks[idx])
            end
            # reduce results into one result
            reduce_fn(results)
        else
            for idx = 1:parallelism
                wait(tasks[idx])
            end
        end
    end
end


## Had lots of help with this from Mark.  Turns out static arrays really
## are a ton faster than anything else.
@inline function distance_squared(dataset_row::D,
                                  centroid::SVector{S, Float64}) where {D, S}
    sum = 0.0
    @fastmath @inbounds @simd for ind = Base.OneTo(S)
        sum += abs2(dataset_row[ind] - centroid[ind])
    end
    sum
end

"""
Iterate through dataset mutable updating distances with the
distance from exactly one centroid.  Then assign the cumulative
summation of distances to scan_distances.
"""
function kmeans_next_centroid(dataset::Array{D,2},
                              centroids::Array{Float64,2},
                              centroid_idx::Int,
                              distances::Array{Float64,1},
                              scan_distances::Array{Float64,1}) where D
    # Convert to julia indexes
    centroid_idx = centroid_idx + 1
    # centroid = SVector{size(centroids,1)}(centroids[:,centroid_idx])
    ncols = size(dataset,1)
    centroid = SVector{ncols,Float64}(centroids[:,centroid_idx])
    Threads.@threads for ind in eachindex(distances)
        col = @view dataset[:, ind]
        dssq = distance_squared(col, centroid)
        @inbounds if (dssq < distances[ind])
            @inbounds distances[ind] = dssq
        end
    end
    # prefix scan
    cumsum!(scan_distances, distances)
end


@inline function centroid_distance_index(dataset_row::AbstractVector{D},
                                         centroid_ary) where D
    mindistance = typemax(Float64)
    minindex = 1
    n_centroids = length(centroid_ary)
    @fastmath @inbounds @simd for centroid_idx in Base.OneTo(n_centroids)
        row_distance = distance_squared(dataset_row, centroid_ary[centroid_idx])
        if (row_distance < mindistance)
            mindistance = row_distance
            minindex = centroid_idx
        end
    end
    return (mindistance,minindex)
end

"""
Assign centroid indexes based on the centroid with the shorted
distance to a given row.  Concurrently calculate and return the 
score from the previous round.  This code shares an array between
the threads sums into it and finally calls sum on the array.

This approach is less efficient and flexible than using an 
indexed_map_reduce.
"""
function assign_centroids(dataset::Array{D,2},
                          centroids::Array{Float64,2},
                          centroid_indexes::Array{Int32,1}) where D
    ncols,nrows = size(dataset)
    ncols,ncentroids = size(centroids)
    scores = zeros(Float64,nrows)
    # score = Threads.Atomic{Float64}(0.0)
    centroid_ary = copy(reinterpret(SVector{ncols,Float64},centroids))
    Threads.@threads for row_idx in 1:nrows
        dataset_row = @view dataset[:,row_idx]
        (mindistance,minindex) = centroid_distance_index(dataset_row, centroid_ary)
        @inbounds scores[row_idx] = mindistance
        # use java zero-based indexing here
        @inbounds centroid_indexes[row_idx] = minindex - 1
    end
    sum(scores)
end

"""
An alternate version of assign centroids to indexes and calculate score
using indexed_map_reduce.  This allows the code below to simply declare
a partial summation on the stack and then sum nthreads partial summations
as a reduction step.
"""
function assign_centroids_imr(dataset::Array{D,2},
                              centroids::Array{Float64,2},
                              centroid_indexes::Array{Int32,1}) where D
    ncols,nrows = size(dataset)
    ncols,ncentroids = size(centroids)
    # score = Threads.Atomic{Float64}(0.0)
    centroid_ary = copy(reinterpret(SVector{ncols,Float64},centroids))
    indexed_map_reduce(nrows,
                       function(s,gl)
                         score = 0.0
                         for row_idx in range(s,length=gl)
                          dataset_row = @view dataset[:,row_idx]
                          (mindistance,minindex) = centroid_distance_index(dataset_row, centroid_ary)
                          # use java zero-based indexing here
                          @inbounds centroid_indexes[row_idx] = minindex - 1
                          score += mindistance
                         end
                         score
                       end,
                       sum)
end


## I attempted to calculate and assign centroids all in Julia and I could not get
## the performance of the combined JVM/Julia algorithm.  Specifically the actual
## multithreaded centroid summation turned out to be either quite a bit slower
## than the jvm centroid summation *or* the compilation time required to get good
## performance was longer than the time it takes to perform the entire kmeans algorithm
## with the test set.  The function below is not used in the actual code.
function assign_calc_centroids(dataset::Array{D,2},
                               centroids::Array{Float64,2},
                               result_centroids::Array{Float64,2}) where D
    ncols,nrows = size(dataset)
    ncols,ncentroids = size(centroids)
    # score = Threads.Atomic{Float64}(0.0)
    centroid_ary = copy(reinterpret(SVector{ncols,Float64},centroids))
    nthreads = Threads.nthreads()
    # Changing these to MArrays makes things a lot faster at the cost of
    # massive compile times.
    new_centroids = Array{Float64,3}(undef, ncols, ncentroids, nthreads)
    row_counts = Array{Int32, 2}(undef, ncentroids, nthreads)
    scores = MArray{Tuple{nthreads}, Float64}(undef)
    new_centroids .= 0.0
    row_counts .= 0
    indexed_map_reduce(nrows,
                       function(row_start::Int, nrows::Int)
                         threadid = Threads.threadid()
                         # Parallelized in outer loop.
                         for row_idx in range(row_start,length=nrows)
                           dataset_row = @view dataset[:,row_idx]
                           (mindistance,minindex) = centroid_distance_index(dataset_row, centroid_ary)
                           # use java zero-based indexing here
                           @inbounds new_centroids[:,minindex,threadid] .+= dataset_row
                           @inbounds row_counts[minindex,threadid] += 1
                           scores[threadid] += mindistance
                         end
                       end)
    sum!(result_centroids, new_centroids)
    cumsum!(row_counts, row_counts, dims=2)
    result_centroids' ./= row_counts[:,nthreads]
    sum(scores)
end

"""
Calculate the cumulative distance of the dataset from a set of centroids.
"""
function score_kmeans(dataset, centroids)
    ncols,nrows = size(dataset)
    ncols,ncentroids = size(centroids)
    score = 0.0
    centroid_ary = copy(reinterpret(SVector{ncols,Float64},centroids))
    indexed_map_reduce(nrows,
                       function(s,gl)
                         score = 0.0
                         for row_idx in range(s,length=gl)
                          dataset_row = @view dataset[:,row_idx]
                          (mindistance,minindex) = centroid_distance_index(dataset_row, centroid_ary)
                          score += mindistance
                         end
                         score
                       end,
                       sum)
end

"""
Order labels such that they are linearly increasing and create
an ordered dataset to match.  Return a tuple of dataset,labels.
"""
function order_data_labels(data::Array{D,2},
                           labels::Array{E,1}) where {D,E}
    indexes = sortperm(labels)
    (data[:,indexes], labels[indexes])
end

"""
Infer the classification of each dataset row assuming a model where
each label gets n centroids.  This algorithm assumes labels are the
integers [0-N) where N is the number of labels.
"""
function per_label_infer(dataset::D,
                         centroids::C,
                         n_labels::Integer,
                         assigned_indexes::AI) where {D,C,AI}
    n_cols,n_rows = size(dataset)
    n_cols,n_centroids = size(centroids)
    n_per_label = n_centroids / n_labels
    centroid_ary = copy(reinterpret(SVector{n_cols,Float64},centroids))
    Threads.@threads for row_idx in Base.OneTo(n_rows)
        dataset_row = @view dataset[:,row_idx]
        (mindistance,minindex) = centroid_distance_index(dataset_row, centroid_ary)
        # minindex needs to be zero based for this to work, and assigned indexes
        # ends up being zero based
        @inbounds assigned_indexes[row_idx] = (minindex-1) ÷ n_per_label
    end
end
