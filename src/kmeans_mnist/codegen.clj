(ns kmeans-mnist.codegen
  (:require [insn.core :as insn]
            [camel-snake-kebab.core :as csk])
  (:import [java.nio ByteBuffer DoubleBuffer]))

(declare eval-insns)


(defmulti insn-dispatch
  "Implement a slightly higher level insn primitive"
  (fn [context insns insn]
    (cond
      (sequential? insn)
      (first insn)
      (number? insn)
      :constant
      (symbol? insn)
      :variable)))


(defmethod insn-dispatch 'foreach
  [context insns insn]
  (let [[_ loop-dtype idx-var n-elem-var & body] insn
        loop-label (keyword (name (gensym "foreach")))
        [var-fn plus-fn]
        (case loop-dtype
          :int32 [(list 'ivar idx-var 0) 'i+]
          :int64 [(list 'lvar idx-var 0) 'l+])
        foreach-insns
        [(list 'mark loop-label)
         ['if< [idx-var n-elem-var]
          ['block
           (concat
            (let [[context insns] (eval-insns context [] body)]
              insns)
            [['setvar! idx-var [plus-fn idx-var 1]]
             ['goto loop-label]])]]]]
    [context (concat insns foreach-insns)]))

(defmethod insn-dispatch :default
  [context insns insn]
  [context (concat insns [insn])])


(defn eval-insns
  [context insns emit-list]
  (reduce (fn [[ctx insns] insn]
            (insn-dispatch ctx insns insn))
          [context []]
          emit-list))


(defn insn-emit-method
  [compile-constants name flags arglist rettype emit-list]
  (let [argname->idx (->> arglist
                          (map-indexed (fn [idx [argtype argname]]
                                         [argname (inc idx)]))
                          (into {}))
        context {:argname->idx argname->idx
                 :compile-constants compile-constants}]
    {:flags flags
     :name name
     :desc (concat (map first arglist) [rettype])
     :emit (eval-insns context [] emit-list)}))


(defn nio-inner-cls-def
  [buftype ncols]
  {:name (symbol (str "kmeans-minst.codegen.NioInnerClsDef_"
                      (.getSimpleName
                       ^Class buftype)
                      "_" ncols))
   :methods [(insn-emit-method
              {'buftype buftype 'ncols ncols}
              :nioInnerLoop
              #{:public :static}
              [[buftype 'dataset]
               [DoubleBuffer 'centroids]
               [DoubleBuffer 'distances]
               [:int32 'start-row]
               [:int32 'nrows]]
              :void
              '[(foreach
                 :int32 local-row-idx nrows
                 (ivar row-idx (i+ local-row-idx start-row))
                 (dvar sum 0.0)
                 (foreach-unroll
                  :int32 10 cent-idx ncols
                  (dvar diff (d- (tget dataset [(i+ (i* row-idx ncols)
                                                    cent-idx)])
                                 (tget centroids [cent-idx])))
                  (setvar! sum (d+ sum (d* diff diff))))
                 (if< [sum (tget distances [row-idx])]
                   (tset! distances [row-idx] sum)))
                (return)])]})
