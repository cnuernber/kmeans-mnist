(ns kmeans-mnist.jl-kmeans
  (:require [libjulia-clj.julia :refer [jl] :as julia]
            [tvm-clj.ast :as ast]
            [tvm-clj.ast.elemwise-op :as ast-op]
            [tvm-clj.schedule :as schedule]
            [tvm-clj.compiler :as compiler]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.argops :as argops]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.tensor :as dtt]
            [tech.v3.parallel.for :as pfor]
            [tech.v3.libs.buffered-image :as bufimg]
            [clojure.java.io :as io])
  (:import [java.util Random]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defonce init* (delay (julia/initialize! {:n-threads -1})))
@init*

(do
  (def loader (jl (slurp (io/resource "kmeans.jl"))))
  (def kmeans-next-centroid (jl "kmeans_next_centroid"))
  (def assign-centroids (jl "assign_centroids"))
  (def score-kmeans (jl "score_kmeans")))


(defn- seed->random
  ^Random [seed]
  (cond
    (number? seed)
    (Random. (int seed))
    (instance? Random seed)
    seed
    (nil? seed)
    (Random.)
    :else
    (errors/throwf "Unrecognized seed type: %s" seed)))

(defmacro distance-squared
  [dataset centroids row-idx centroid-idx ncols]
  `(double
    (loop [col-idx# 0
           sum# 0.0]
      (if (< col-idx# ~ncols)
        (let [diff# (- (.ndReadDouble ~dataset ~row-idx col-idx#)
                       (.ndReadDouble ~centroids ~centroid-idx col-idx#))]
          (recur (unchecked-inc col-idx#) (+ sum# (* diff# diff#))))
        sum#))))


(defn jvm-next-centroid
  [dataset centroids idx distances scan-distances]
  (let [dataset (dtt/as-tensor dataset)
        centroids (dtt/as-tensor centroids)
        distances (dtype/->buffer distances)
        scan-distances (dtype/->buffer scan-distances)
        centroid-idx (long idx)
        [nrows ncols] (dtype/shape dataset)
        nrows (long nrows)
        ncols (long ncols)]
    (pfor/parallel-for
     row-idx nrows
     (let [sum (distance-squared dataset centroids row-idx centroid-idx ncols)]
       (when (< sum (.readDouble distances row-idx))
         (.writeDouble distances row-idx sum))))
    (.writeDouble scan-distances 0 (.readDouble distances 0))
    (loop [idx 1]
      (when (< idx nrows)
        (.writeDouble scan-distances idx
                        (+ (.readDouble scan-distances (unchecked-dec idx))
                           (.readDouble distances idx)))
        (recur (unchecked-inc idx))))))


(defn- tvm-dist-sum-algo
  "Update the distances with values from the new centroid and produce a cumulative
  summation vector we can binary search through.  This uses a special form where we
  just add in the new centroid to our existing distance vector and recalculate our
  cumulative summation vector.  This is similar to the distance methods below except
  we only calculate one centroid at a time."
  [n-cols dataset-datatype]
  (let [n-centroids (ast/variable "n_centroids")
        n-rows (ast/variable "nrows")
        n-cols (ast-op/const n-cols :int32)
        center-idx (ast/variable "center-idx")
        ;;The distance calculation is the only real issue here.
        ;;Everything else, sort, etc. is pretty quick and sorting
        centroids (ast/placeholder [n-centroids n-cols] "centroids" :dtype :float64)
        dataset (ast/placeholder [n-rows n-cols] "dataset" :dtype dataset-datatype)
        ;;distances are doubles so summation is in double space
        distances (ast/placeholder [n-rows] "distances" :dtype :float64)
        ;;Single centroid squared distance calculation
        squared-diff (-> (ast/compute
                          [n-rows n-cols] "squared-diff" nil
                          [row-idx col-idx]
                          (ast/tvm-let
                           [row-elem (-> (ast/tget dataset [row-idx col-idx])
                                         (ast-op/cast :float64))
                            center-elem (ast/tget centroids [center-idx col-idx])
                            diff (ast-op/- row-elem center-elem)]
                           (ast-op/* diff diff)))
                         (ast/first-output))
        squared-dist (-> (ast/compute
                          [n-rows] "expanded-distances" nil
                          [row-idx]
                          (ast/commutative-reduce
                           [:+ :float64]
                           [{:domain [0 n-cols] :name "col-idx"}]
                           [(fn [col-idx] (ast/tget squared-diff [row-idx col-idx]))]))
                         (ast/first-output))
        ;;Aggregate previous distances, new distance into result.
        mindistances (-> (ast/compute
                          [n-rows] "mindistances" nil
                          [row-idx]
                          (ast/tvm-let
                           [prev-dist (ast-op/select (ast-op/eq center-idx 0)
                                                     (ast-op/max-value :float64)
                                                     (ast/tget distances [row-idx]))
                            cur-dist (-> (ast/tget squared-dist [row-idx])
                                         (ast-op/cast :float64))]
                           (ast-op/min cur-dist prev-dist)))
                         (ast/first-output))

        ;;Produce the cumulative summation vector.  For this system, we define the
        ;;algorithm in terms of timesteps.  We should have n-rows timesteps for our
        ;;algorithm state.
        scan-state (ast/placeholder [n-rows] "scan_state" :dtype :float64)

        scan-result (-> (ast/scan
                         ;;First compute op sets up initial state at timestep 0.  This
                         ;;could setup an arbitrary amount of initial state.
                         (ast/compute [1] "init" nil
                                      [row-idx]
                                      (ast/tget mindistances [row-idx]))
                         ;;Next we describe our recursive update in terms of reading
                         ;;from the state vector at previous timesteps and we can read
                         ;;from anything else at the current timestep.
                         (ast/compute [n-rows] "update" nil
                                      [ts-idx]
                                      (ast-op/+
                                       ;;grab stage from ts-1
                                       (ast/tget scan-state [(ast-op/- ts-idx (int 1))])
                                       ;;add to incoming values
                                       (ast/tget mindistances [ts-idx])))
                         ;;State of scan algorithm.  Must have enough dimensions for
                         ;;each timestep as well as result
                         scan-state
                         ;;incoming values
                         [mindistances]
                         {:name "distance_scan"})
                        (ast/first-output))
        schedule (-> (schedule/create-schedule scan-result)
                     (schedule/inline-op squared-diff squared-dist -1)
                     (schedule/inline-op squared-dist mindistances 0)
                     (schedule/parallelize-axis mindistances 0))]
    {:arguments [dataset centroids center-idx distances mindistances scan-result]
     :schedule schedule}))


(def ^:private tvm-next-centroid*
  (delay
   (let [raw-tvm-fn (compiler/ir->fn (tvm-dist-sum-algo 3 :uint8) "dist_sum")]
     (fn [dataset centroids idx distances scan-distances]
       (let [distances (dtt/as-tensor distances)]
         (raw-tvm-fn (dtt/as-tensor dataset)
                     (dtt/as-tensor centroids)
                     idx
                     distances
                     distances
                     (dtt/as-tensor scan-distances)))))))


(defn- choose-centroids++
  "Implementation of the kmeans++ center choosing algorithm."
  [dataset n-centroids {:keys [seed
                               distance-fn]}]
  ;;Note julia is col major but datatype is row major
  (let [[n-rows n-cols] (dtype/shape dataset)
        ;;Remember julia is column major, so shape arguments are reversed
        centroids (julia/new-array [n-cols n-centroids] :float64)]
    (let [centroid-tens (dtt/as-tensor centroids)
          ds-tens (dtt/as-tensor dataset)
          random (seed->random seed)
          n-rows (long n-rows)
          distances (julia/new-array [n-rows] :float64)
          _ (dtype/set-constant! distances Double/MAX_VALUE)
          scan-distances (julia/new-array [n-rows] :float64)
          scan-dist-tens (dtt/as-tensor scan-distances)
          initial-seed-idx (.nextInt random (int n-rows))
          _ (dtt/mset! centroid-tens 0 (dtt/mget ds-tens initial-seed-idx))
          n-centroids (long n-centroids)
          last-idx (dec n-rows)]
      (dotimes [idx (dec n-centroids)]
        (distance-fn dataset centroids idx distances scan-distances)
        (let [next-flt (.nextDouble ^Random random)
              n-rows (dtype/ecount distances)
              distance-sum (double (scan-dist-tens (dec n-rows)))
              target-amt (* next-flt distance-sum)
              next-center-idx (min last-idx
                                   ;;You want the one just *after* where you could
                                   ;;safely insert the distance as the next distance is
                                   ;;likely much larger than the current distance and
                                   ;;thus your probability of getting a vector that
                                   ;;that is a large distance away than any known
                                   ;;vectors is higher
                                   (inc (argops/binary-search scan-dist-tens
                                                              target-amt)))]
          (dtt/mset! centroid-tens (inc idx) (dtt/mget ds-tens next-center-idx))))
      centroids)))


(defn- jvm-assign-centroids
  [dataset centroids centroid-indexes]
  (let [dataset (dtt/as-tensor dataset)
        centroids (dtt/as-tensor centroids)
        ;;Use 1d indexing
        centroid-indexes (dtype/->buffer centroid-indexes)
        [nrows,ncols] (dtype/shape dataset)
        [ncentroids,ncols] (dtype/shape centroids)
        nrows (long nrows)
        ncols (long ncols)
        ncentroids (long ncentroids)]
    (pfor/indexed-map-reduce
     nrows
     (fn [^long start-idx ^long group-len]
       (let [end-row (+ start-idx group-len)]
         (loop [row-idx start-idx
                sum 0.0]
           (if (< row-idx end-row)
             (let [[dist cent-idx]
                   (loop [cent-idx 0
                          mindist Double/MAX_VALUE
                          minidx 0]
                     (if (< cent-idx ncentroids)
                       (let [newdist (distance-squared dataset centroids row-idx
                                                       cent-idx ncols)
                             minidx (if (< newdist mindist)
                                      cent-idx
                                      minidx)
                             mindist (if (< newdist mindist)
                                       newdist
                                       mindist)]
                         (recur (unchecked-inc cent-idx) mindist minidx))
                       [mindist minidx]))]
               (.writeLong centroid-indexes row-idx cent-idx)
               (recur (unchecked-inc row-idx) (+ sum (double dist))))
             sum))))
     (partial reduce +))))


(comment
  (do
    (def src-image (bufimg/load "data/jen.jpg"))
    (def img-height (first (dtype/shape src-image)))
    (def img-width (second (dtype/shape src-image)))
    (def nrows (* img-width img-height))
    (def ncols (/ (dtype/ecount src-image)
                  nrows))
    (def dataset (-> (dtt/reshape src-image [nrows ncols])
                     (julia/->array)))
    (def centroid-indexes (-> (dtt/new-tensor [nrows] :datatype :int32)
                              (julia/->array)))
    )

  (def centroids (time (choose-centroids++
                        dataset 5 {:seed 5 :distance-fn kmeans-next-centroid})))
  ;; 285ms
  (def centroids (time (choose-centroids++
                        dataset 5 {:seed 5 :distance-fn jvm-next-centroid})))
  ;;1163ms
  (def centroids (time (choose-centroids++
                        dataset 5 {:seed 5 :distance-fn @tvm-next-centroid*})))
  ;; 345ms
  (def score (time (assign-centroids dataset centroids centroid-indexes)))
  ;;1789ms
  (def score (time (jvm-assign-centroids dataset centroids centroid-indexes)))
  ;; 775ms
  )
