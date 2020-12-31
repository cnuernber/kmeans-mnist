@inline function distance_squared(dataset::Array{UInt8,2},
                                  centroids::Array{Float64,2},
                                  ncols::Int,
                                  row_idx::Int,
                                  centroid_idx::Int)
    sum = 0.0
    for idx in 1:ncols
        diff = convert(Float64, dataset[idx, row_idx])-centroids[idx, centroid_idx]
        sum += diff * diff
    end
    return sum
end


@inbounds function kmeans_next_centroid(dataset::Array{UInt8,2},
                              centroids::Array{Float64,2},
                              centroid_idx::Int,
                              distances::Array{Float64,1},
                              scan_distances::Array{Float64,1})
    # expect zero-based indexing upon incoming data
    centroid_idx = centroid_idx + 1
    ncols,nrows = size(dataset)
    ncols,ncentroids = size(centroids)
     Threads.@threads for row_idx in 1:nrows
        dssq = distance_squared(dataset, centroids, ncols, row_idx, centroid_idx)
        if (dssq < distances[row_idx])
            distances[row_idx] = dssq
        end
    end
    # prefix scan
    scan_distances[1] = distances[1]
    for row_idx in 2:nrows
        scan_distances[row_idx] = scan_distances[row_idx - 1] + distances[row_idx]
    end
end


function centroid_distance_index(row, centroids, ncentroids, ncols)
    mindistance = typemax(Float64)
    minindex = 1
    for centroid_idx in 1:ncentroids
        row_distance = distance_squared(row, centroids[:,centroid_idx])
        if (row_distance < mindistance)
            mindistance = row_distance
            minindex = centroid_idx
        end
    end
    return (mindistance,minindex)
end


function assign_centroids(dataset, centroids, centroid_indexes)
    ncols,nrows = size(dataset)
    ncols,ncentroids = size(centroids)
    score = 0.0
    Threads.@threads for row_idx in 1:nrows
        row = dataset[:,row_idx]
        (mindistance,minindex) = centroid_distance_index(row, centroids,
                                                         ncentroids, ncols)
        score += mindistance
        # use java zero-based indexing here
        centroid_indexes[row_idx] = minindex - 1
    end
    score
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
