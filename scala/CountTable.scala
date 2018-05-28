import scala.collection.mutable

class CountTable[T](implicit ord: Ordering[T]) extends Iterable[(T, Double)] {

  private[this] val counts = mutable.SortedMap.empty[T, Double]

  def putZero(key: T): Unit = {
    counts(key) = 0.0
  }

  def addKey(key: T, n: Double = 1.0): Unit = if (n > 0.0) {
    counts(key) = counts.getOrElse(key, 0.0) + n
  }

  def iterator: Iterator[(T, Double)] = counts.iterator

  def clear(): Unit = counts.clear()
}

object CountTable {

  def sortCount[T](iter: Iterator[(T, Double)],
                   cut: Double = 0.0,
                   backoff: T => Option[T] = (_:T) => None)
                  (implicit ord: Ordering[T]): Iterator[(T, Double)] = {

    val pq = mutable.PriorityQueue.empty[(T, Double)](
      Ordering.Tuple2(Ordering.Double, ord.reverse).on[(T, Double)](x => (x._2, x._1)))
    val tab = new CountTable[T]()(ord)

    for (x <- iter) {
      if (x._2 < cut) backoff(x._1) match {
        case Some(y) =>
          tab.addKey(y, x._2)
        case None => //PASS
      }
      else pq.enqueue(x)
    }
    for (x <- tab) pq.enqueue(x)
    tab.clear()

    new Iterator[(T, Double)] {
      def hasNext: Boolean = pq.nonEmpty
      def next(): (T, Double) = pq.dequeue()
    }
  }

  def topCount[T](iter: Iterator[(T, Double)], topk: Int,
                  backoff: T => Option[T] = (_:T) => None)
                 (implicit ord: Ordering[T]): Iterator[(T, Double)] = {

    val pq = mutable.PriorityQueue.empty[(T, Double)](
      Ordering.Tuple2(Ordering.Double.reverse, ord).on[(T, Double)](x => (x._2, x._1)))
    pq.sizeHint(topk)

    val tab = new CountTable[T]()(ord)
    for (x <- iter) {
      pq.enqueue(x)
      while (pq.size + tab.size > topk) {
        val (k, v) = pq.dequeue()
        backoff(k) match {
          case Some(y) =>
            tab.addKey(y, v)
          case None => //PASS
        }
      }
    }
    for (x <- tab) pq.enqueue(x)
    tab.clear()

    pq.dequeueAll.reverseIterator
  }

  def mergeCount[T](iters: Iterable[Iterator[(T, Double)]])(implicit ord: Ordering[T]): Iterator[(T, Double)] =
    new Iterator[(T, Double)] {
      private[this] val key_pool = mutable.Map.empty[T, (Double, List[Iterator[(T, Double)]])]
      private[this] val key_pq = mutable.PriorityQueue.empty[T](ord.reverse)

      private[this] def process(iter: Iterator[(T, Double)]): Unit =
        if (iter.hasNext) {
          val (key, num) = iter.next()
          key_pool.get(key) match {
            case Some((n, its)) =>
              key_pool(key) = (num + n, iter :: its)
            case None =>
              key_pq.enqueue(key)
              key_pool(key) = (num, iter :: Nil)
          }
        }

      iters.foreach(process)

      def hasNext: Boolean = key_pq.nonEmpty

      def next(): (T, Double) = {
        val ret_key = key_pq.dequeue()
        val (ret_num, its) = key_pool.remove(ret_key).get
        its.foreach(process)
        (ret_key, ret_num)
      }
    }
}
