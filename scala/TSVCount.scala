import java.io.PrintWriter

object TSVCount {

  def read(lines: Iterator[String]): Iterator[(String, Double)] = {
    for (line <- lines) yield {
      val sp = line.split("\t")
      (sp(0), sp(1).toDouble)
    }
  }

  def write(fn: String, count: Iterator[(String, Double)]): Unit = {
    val pw = new PrintWriter(fn)
    for ((s, num) <- count) {
      pw.println(s + "\t" + num)
    }
    pw.close()
  }
}
