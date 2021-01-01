using StaticArrays

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
    ignored,n_centroids = size(centroid_ary)
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
    score = Threads.Atomic{Float64}(0.0)
    centroid_ary = reinterpret(SVector{ncols,Float64},centroids)
    Threads.@threads for row_idx in 1:nrows
        dataset_row = @view dataset[:,row_idx]
        (mindistance,minindex) = centroid_distance_index(dataset_row, centroid_ary)
        Threads.atomic_add!(score,mindistance)
        # use java zero-based indexing here
        @inbounds centroid_indexes[row_idx] = minindex - 1
    end
    score[]
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
