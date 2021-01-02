(ns kmeans-mnist.jl-kmeans
  (:require [libjulia-clj.julia :refer [jl] :as julia]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.argops :as argops]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.datatype.functional :as dfn]
            [tech.v3.tensor :as dtt]
            [tech.v3.parallel.for :as pfor]
            [tech.v3.datatype.reductions :as reductions]
            [tech.v3.libs.buffered-image :as bufimg]
            [clojure.java.io :as io])
  (:import [java.util Random HashMap]
           [java.util.function BiFunction BiConsumer]
           [tech.v3.datatype ArrayHelpers IndexReduction
            IndexReduction$IndexedBiFunction]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defonce init* (delay (julia/initialize! {:n-threads -1})))
@init*

(do
  (def loader (jl (slurp (io/resource "kmeans.jl"))))
  (def kmeans-next-centroid (jl "kmeans_next_centroid"))
  (def assign-centroids (jl "assign_centroids"))
  (def assign-centroids-imr (jl "assign_centroids_imr"))
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


(defrecord AggReduceContext [^doubles center
                             ^longs n-rows])


(defn- new-center-reduction
  ^IndexReduction [dataset]
  (let [[_nrows n-cols] (dtype/shape dataset)
        dataset (dtype/->buffer dataset)
        make-reduce-context #(->AggReduceContext (double-array n-cols)
                                                 (long-array 1))
        n-cols (long n-cols)]
    (reify IndexReduction
      (reduceIndex [this batch ctx row-idx]
        (let [^AggReduceContext ctx (or ctx (make-reduce-context))
              row-off (* n-cols row-idx)]
          (dotimes [col-idx n-cols]
            (ArrayHelpers/accumPlus ^doubles (.center ctx) col-idx
                                    (.readDouble dataset (+ row-off col-idx))))
          (ArrayHelpers/accumPlus ^longs (.n-rows ctx) 0 1)
          ctx))
      (reduceReductions [this lhsCtx rhsCtx]
        (let [^AggReduceContext lhsCtx lhsCtx
              ^AggReduceContext rhsCtx rhsCtx]
          (dotimes [col-idx n-cols]
            (ArrayHelpers/accumPlus ^doubles (.center lhsCtx) col-idx
                                    (aget ^doubles (.center rhsCtx) col-idx)))
          (ArrayHelpers/accumPlus ^longs (.n-rows lhsCtx) 0
                                  (aget ^longs (.n-rows rhsCtx) 0))
          lhsCtx)))))


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
        ncentroids (long ncentroids)
        reducer (new-center-reduction dataset)
        merge-bifn (reify BiFunction
                     (apply [this lhs rhs]
                       (.reduceReductions reducer lhs rhs)))]
    (let [[score center-map]
          (pfor/indexed-map-reduce
           nrows
           (fn [^long start-idx ^long group-len]
             (let [end-row (+ start-idx group-len)
                   center-map (java.util.HashMap.)
                   bifn (IndexReduction$IndexedBiFunction. reducer nil)]
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
                     (.setIndex bifn row-idx)
                     (.compute center-map cent-idx bifn)
                     (recur (unchecked-inc row-idx) (+ sum (double dist))))
                   [sum center-map]))))
           (partial reduce (fn [[rsum ^HashMap rcenter-map] [sum ^HashMap center-map]]
                             (.forEach center-map
                                       (reify BiConsumer
                                         (accept [this k v]
                                           (.merge rcenter-map k v merge-bifn))))
                             [(+ (double rsum) (double sum)) rcenter-map])))
          new-centroids (dtt/new-tensor (dtype/shape centroids) :datatype :float64)]
      (.forEach ^HashMap center-map
                (reify BiConsumer
                  (accept [this center-idx reduce-context]
                    (let [^doubles center (:center reduce-context)
                          ^longs n-rows (:n-rows reduce-context)]
                      (dtt/mset! new-centroids center-idx
                                 (dfn// center (aget n-rows 0)))))))
      [score new-centroids])))


(defn jvm-assign-centers-from-centroid-indexes
  [dataset center-indexes]
  (let [reducer (new-center-reduction dataset)
        center-map (reductions/ordered-group-by-reduce reducer nil center-indexes)
        new-centroids (dtt/new-tensor [(count center-map)
                                     (second (dtype/shape dataset))])]
    (.forEach ^HashMap center-map
              (reify BiConsumer
                (accept [this center-idx reduce-context]
                  (let [^doubles center (:center reduce-context)
                        ^longs n-rows (:n-rows reduce-context)]
                    (dtt/mset! new-centroids center-idx
                               (dfn// center (aget n-rows 0)))))))
    new-centroids))


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
  ;; 103ms
  (def score (time (assign-centroids-imr dataset centroids centroid-indexes)))
  ;; 103ms
  (def score (time (jvm-assign-centroids dataset centroids centroid-indexes)))
  ;; 775ms

  (def jvm-centroids
    (time (jvm-assign-centers-from-centroid-indexes dataset centroid-indexes)))
  ;; 169ms
  )
