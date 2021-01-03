using StaticArrays


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


@inline function distance_squared(dataset_row::AbstractVector{D},
                                  centroid::SVector{S, Float64}) where D where S
    sum = 0.0
    @fastmath @inbounds @simd for ind = Base.OneTo(S)
        sum += abs2(dataset_row[ind] - centroid[ind])
    end
    sum
end

function kmeans_next_centroid(dataset::Array{D,2},
                              centroids::Array{Float64,2},
                              centroid_idx::Int,
                              distances::Array{Float64,1},
                              scan_distances::Array{Float64,1}) where D
    # Convert to julia indexes
    centroid_idx = centroid_idx + 1
    centroid = SVector{size(centroids,1)}(centroids[:,centroid_idx])
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


function reduce_calc_centroids(result_ary::Array{Tuple{Float64,Array{Float64,2},Array{Int32,1}},1})
    score, new_centroids, centroid_row_counts = result_ary[1]
end


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
