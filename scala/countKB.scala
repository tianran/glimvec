
object countKB {

  def main(args: Array[String]): Unit = {
    val Array(train_fn, out_entity_fn, out_relation_fn) = args

    val entity_count = new CountTable[String]()
    val relation_count = new CountTable[String]()

    val file = io.Source.fromFile(train_fn)
    for (line <- file.getLines()) {
      val Array(head, relation, tail) = line.split("\t")
      entity_count.addKey(head)
      relation_count.addKey(relation)
      entity_count.addKey(tail)
    }
    file.close()

    TSVCount.write(out_entity_fn, CountTable.sortCount(entity_count.iterator))
    TSVCount.write(out_relation_fn, CountTable.sortCount(relation_count.iterator))
  }
}
