using StaticArrays

function indexed_map_reduce(num_iters::Int,
                            index_map_fn,
                            reduce_fn)
    parallelism = Threads.nthreads()
    if (num_iters < (2 * parallelism)
        || ccall(:jl_in_threaded_region, Cint, ()) != 0
        || parallelism <= 1)
        reduce_fn([index_map_fn(1, num_iters)])
    else
        group_len, rem = divrem(num_iters, parallelism)
        # Launch all the tasks
        tasks = map(function (idx::Int)
                      # Zero based seems simpler here
                      idx = idx - 1
                      any_leftover = idx < rem
                      start_idx = idx * group_len + 1
                      if any_leftover
                        start_idx += idx
                      else
                        start_idx += rem
                      end
                      llen = group_len
                      if any_leftover
                        llen += 1
                      end
                    Threads.@spawn begin
                                     ccall(:jl_enter_threaded_region, Cvoid, ())
                                     retval = index_map_fn(start_idx, llen)
                                     ccall(:jl_exit_threaded_region, Cvoid, ())
                                     retval
                                   end
                    end,
                    Base.OneTo(parallelism))
        # reduce result
        reduce_fn(map(Threads.fetch, tasks))
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


function score_kmeans(dataset, centroids)
    ncols,nrows = size(dataset)
    ncols,ncentroids = size(centroids)
    score = 0.0
    Threads.@threads for row_idx in 1:nrows
        row = dataset[:,row_idx]
        (mindistance,minindex) = centroid_distance_index(row, centroids,
                                                         ncentroids, ncols)
        score += mindistance
    end
    score
end
