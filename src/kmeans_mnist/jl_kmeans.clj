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
            [clojure.java.io :as io]
            [clojure.tools.logging :as log])
  (:import [java.util Random HashMap]
           [java.util.function BiFunction BiConsumer]
           [tech.v3.datatype ArrayHelpers IndexReduction
            IndexReduction$IndexedBiFunction]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defonce init* (delay (julia/initialize! {:n-threads -1
                                          :optimization-level 3})))
@init*

(julia/set-julia-gc-root-log-level! :info)

(do
  (def loader (jl (slurp (io/resource "kmeans.jl"))))
  (def kmeans-next-centroid (jl "kmeans_next_centroid"))
  (def assign-centroids (jl "assign_centroids"))
  (def assign-centroids-imr (jl "assign_centroids_imr"))
  (def assign-calc-centroids (jl "assign_calc_centroids"))
  (def score-kmeans (jl "score_kmeans"))
  (def order-data-labels (jl "order_data_labels"))
  (def per-label-infer (jl "per_label_infer"))
  )


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
    (julia/with-stack-context
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
        centroids))))


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
  [dataset n-centers center-indexes]
  (let [reducer (new-center-reduction dataset)
        center-map (reductions/ordered-group-by-reduce reducer nil center-indexes)
        new-centroids (dtt/new-tensor [n-centers (second (dtype/shape dataset))])]
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
    (time (do
            (assign-centroids-imr dataset centroids centroid-indexes)
            (jvm-assign-centers-from-centroid-indexes dataset
                                                      (count centroids)
                                                      centroid-indexes))))
  ;; 200ms

  (def jl-centroids (time (assign-calc-centroids dataset centroids new-centroids)))
  ;; 400ms -> 1700ms, varying
  (jl "ccall(:jl_in_threaded_region, Cint, ())")
  )


(defn kmeans++
  "Find K cluster centroids via kmeans++ center initialization
  followed by Lloyds algorithm.
  Dataset must be a matrix (2d tensor).

  * `dataset` - 2d matrix of numeric datatype.
  * `n-centroids` - How many centroids to find.

  Returns map of:

  * `:centroids` - 2d tensor of double centroids
  * `:centroid-indexes` - 1d integer vector of assigned center indexes.
  * `:iteration-scores` - n-iters+1 length array of mean squared error scores container
    the scores from centroid assigned up to the score when the algorithm
    terminates.

  Options:

  * `:minimal-improvement-threshold` - defaults to 0.01 - algorithm terminates if
     (1.0 - error(n-1)/error(n-2)) < error-diff-threshold.  When Zero means algorithm will
     always train to max-iters.
  * `:n-iters` - defaults to 100 - Max number of iterations, algorithm terminates
     if `(>= iter-idx n-iters).
  * `:rand-seed` - integer or implementation of `java.util.Random`."
  [dataset n-centroids & [{:keys [n-iters rand-seed
                                  minimal-improvement-threshold]
                           :or {minimal-improvement-threshold 0.01}
                           :as options}]]
  (errors/when-not-error
   (== 2 (dtype/ecount (dtype/shape dataset)))
   "Dataset must be a matrix of rank 2")
  (let [[n-rows n-cols] (dtype/shape dataset)
        ds-dtype (dtype/elemwise-datatype dataset)
        n-iters (long (or n-iters 100))
        minimal-improvement-threshold (double (or minimal-improvement-threshold
                                                  0.011))]
    (log/infof "Choosing n-centroids %d with %f improvement threshold and max %d iters"
               n-centroids minimal-improvement-threshold n-iters)
    (julia/with-stack-context
      (let [dataset (julia/->array dataset)
            centroids (if (number? n-centroids)
                        (choose-centroids++ dataset n-centroids
                                            {:seed rand-seed
                                             :distance-fn kmeans-next-centroid})
                        (do
                          (errors/when-not-error
                              (== 2 (count (dtype/shape n-centroids)))
                            "Centroids must be rank 2")
                          (julia/->array n-centroids)))
            centroid-indexes (julia/->array (dtt/new-tensor [n-rows] :datatype :int32))
            dec-n-iters (dec n-iters)
            n-rows (long n-rows)
            scores (if-not (== 0 n-iters)
                     (loop [iter-idx 0
                            last-score 0.0
                            scores []]
                       (let [score (assign-centroids-imr dataset centroids
                                                         centroid-indexes)
                             new-centroids (jvm-assign-centers-from-centroid-indexes
                                            dataset n-centroids centroid-indexes)
                             score (/ (double score) n-rows)
                             rel-score (if-not (== 0.0 last-score)
                                         (- 1.0 (/ score last-score))
                                         1.0)]
                         (dtype/copy! new-centroids centroids)
                         (log/infof "Iteration %d out of %d - relative improvement %f->%f=%f"
                                    iter-idx n-iters last-score score rel-score)
                         (if (and (< iter-idx dec-n-iters)
                                  (not= 0.0 score)
                                  (> rel-score minimal-improvement-threshold))
                           (recur (unchecked-inc iter-idx) score (conj scores score))
                           scores)))
                     [])
            final-score (score-kmeans dataset centroids)]
        ;;Clone data back into jvm land to escape the resource context
        {:centroids (dtt/clone centroids)
         :centroid-indexes (dtt/clone centroid-indexes)
         :iteration-scores (vec (concat scores [(/ (double final-score)
                                                   (double n-rows))]))}))))


(comment
  (do
    (def src-image (bufimg/load "data/jen.jpg"))
    (def img-height (first (dtype/shape src-image)))
    (def img-width (second (dtype/shape src-image)))
    (def nrows (* img-width img-height))
    (def ncols (/ (dtype/ecount src-image) nrows))
    (def dataset (-> (dtt/reshape src-image [nrows ncols]))))

  (def img-data (time (kmeans++ dataset 5 {:rand-seed 5})))
  ;;2716ms
  )


(defn- concatenate-results
  "Given a sequence of maps, return one result map with
  tensors with one extra dimension.  Works when every result has the
  same length."
  [result-seq]
  (when (seq result-seq)
    (->> (first result-seq)
         (map (fn [[k v]]
                [k (dtt/->tensor (mapv k result-seq)
                                 :datatype (dtype/elemwise-datatype v))]))
         (into {}))))


(defn train-per-label
  "Given a dataset along with per-row integer labels, train N per-label
  kmeans centroids returning a model which you can use can use with predict-per-label."
  [data labels n-per-label & [{:keys [input-ordered?]
                               :as options}]]
  (julia/with-stack-context
    (when-not (empty? labels)
      ;;Organize data per-label
      (let [n-per-label (long n-per-label)
            ds-dtype (dtype/elemwise-datatype data)
            [data labels] (if input-ordered?
                            [(julia/->array data) (julia/->array labels)]
                            ;;Order data and labels by increasing index
                            (order-data-labels (julia/->array data)
                                               (julia/->array labels)))
            [n-rows n-cols] (dtype/shape data)
            labels (->> (argops/arggroup labels)
                        (into {})
                        (sort-by first)
                        ;;arggroup be default uses an 'ordered' algorithm that guarantees
                        ;;the result index list is ordered.
                        (mapv (fn [[label idx-list]]
                                [label
                                 [(first idx-list) (last idx-list)]])))
            n-labels (count labels)]
        (->> labels
             (map (fn [[label [^long idx-start ^long past-idx-end]]]
                    ;;Tensor selection from contiguous data of a range with an increment of 1
                    ;;is guaranteed to produce contiguous data
                    (log/infof "Training centroids for label %s" label)
                    (let [{:keys [centroids centroid-indexes iteration-scores]}
                          (-> (dtt/select data (range idx-start past-idx-end))
                              (kmeans++ n-per-label options))]
                      {:centroids centroids
                       :labels label
                       :iteration-scores (last iteration-scores)})))
             (concatenate-results)
             (merge {:kmeans-type :n-per-label}))))))


(defn predict-per-label
  "Using a per-label `model`, find the nearest centroid to each row
  and return a 1d tensor of the predicted label.

  Returns:

  * `:label-indexes` - int32 assigned indexes for each row in the dataset."
  [dataset model]
  (julia/with-stack-context
    (let [{:keys [centroids labels]} model
          [n-labels n-per-label n-cols] (dtype/shape centroids)]
      ;;Eventually we will have a resource context here
      (let [[n-labels n-per-label n-cols] (dtype/shape centroids)
            [n-rows n-data-cols] (dtype/shape dataset)
            _ (errors/when-not-errorf
                  (= n-cols n-data-cols)
                "Data (%d), model (%d) have different feature counts"
                n-data-cols n-cols)
            dataset (julia/->array dataset)
            n-centroids (* (long n-labels)
                           (long n-per-label))
            centroids (-> (dtt/reshape centroids [n-centroids n-cols])
                          (julia/->array))
            indexes (julia/new-array [n-rows] :int32)]
        (per-label-infer dataset centroids n-labels indexes)
        {:label-indexes (dtype/clone indexes)}))))
